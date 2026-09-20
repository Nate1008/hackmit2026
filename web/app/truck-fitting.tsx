"use client";

import { Edges, OrbitControls, TransformControls, useGLTF } from "@react-three/drei";
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const API_BASE =
  process.env.NEXT_PUBLIC_SHAPER_API_URL?.replace(/\/$/, "") ||
  "/api/shaper";

declare global {
  interface Window {
    google?: any;
    gm_authFailure?: () => void;
    __honkpackGoogleMapsReady?: () => void;
    __honkpackGoogleMapsPromise?: Promise<any>;
  }
}

export type PackingItem = {
  id: string;
  name: string;
  width: number;
  height: number;
  depth: number;
  color: string;
  url: string;
};

type Dimensions = { width: number; height: number; depth: number };
type CargoProfile = { payloadKg: number; pricePerHour: number };
type TruckModel = {
  id: string;
  name: string;
  dimensions: Dimensions;
  modelUrl: string;
  cargoOffset: [number, number, number];
  cargo: CargoProfile;
};
type MeasuredItem = PackingItem & { boxYaw: number; meshScale: number };
type PlacedItem = MeasuredItem & {
  packedWidth: number;
  packedHeight: number;
  packedDepth: number;
  packedRotationY: number;
  position: [number, number, number];
};
type ItemTransform = { position: [number, number, number]; rotationY: number };
type EditMode = "translate" | "rotate";
type CameraView = "perspective" | "top" | "side";
type PackingStrategy = "balanced" | "weight" | "footprint";
type RawMeasurement = Dimensions & { yaw: number };
type WeightEstimate = { kg: number; basis: string };

export const TRUCK_MODELS: TruckModel[] = [
  { id: "suv", name: "Toyota RAV4", dimensions: { width: 0.98, height: 0.66, depth: 0.90 }, modelUrl: "/vehicles/suv.glb", cargoOffset: [0, 0.40, -0.59], cargo: { payloadKg: 450, pricePerHour: 58 } },
  { id: "pickup", name: "Ford F-150", dimensions: { width: 1.06, height: 0.43, depth: 1.20 }, modelUrl: "/vehicles/truck.glb", cargoOffset: [0, 0.39, -0.77], cargo: { payloadKg: 700, pricePerHour: 72 } },
  { id: "cargo-van", name: "Ford Transit 250", dimensions: { width: 1.08, height: 0.94, depth: 1.42 }, modelUrl: "/vehicles/van.glb", cargoOffset: [0, 0.30, -0.31], cargo: { payloadKg: 900, pricePerHour: 88 } },
  { id: "box-truck", name: "Isuzu NPR-HD", dimensions: { width: 1.14, height: 1.10, depth: 1.84 }, modelUrl: "/vehicles/delivery.glb", cargoOffset: [0, 0.34, -0.57], cargo: { payloadKg: 1800, pricePerHour: 124 } },
];

TRUCK_MODELS.forEach((truck) => useGLTF.preload(truck.modelUrl));

const EPSILON = 1e-7;
const measurementCache = new Map<string, Promise<RawMeasurement>>();
const WEIGHT_PRIORS = [
  { terms: ["sofa", "couch"], base: 14, density: 52, min: 18, max: 95, basis: "upholstered furniture" },
  { terms: ["chair", "stool"], base: 2.5, density: 35, min: 3, max: 32, basis: "seating" },
  { terms: ["table", "desk"], base: 6, density: 58, min: 7, max: 110, basis: "table furniture" },
  { terms: ["mattress"], base: 3, density: 24, min: 6, max: 48, basis: "mattress" },
  { terms: ["bed", "dresser", "cabinet", "wardrobe"], base: 12, density: 62, min: 16, max: 150, basis: "case furniture" },
  { terms: ["fridge", "refrigerator", "washer", "dryer", "appliance"], base: 24, density: 120, min: 35, max: 180, basis: "large appliance" },
  { terms: ["tv", "television", "monitor"], base: 3, density: 52, min: 4, max: 45, basis: "display electronics" },
  { terms: ["suitcase", "luggage", "bag"], base: 2, density: 82, min: 2.5, max: 34, basis: "packed luggage" },
  { terms: ["box", "carton", "crate"], base: 0.8, density: 105, min: 1, max: 36, basis: "packed moving box" },
];

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function estimateWeight(item: MeasuredItem): WeightEstimate {
  const label = item.name.toLowerCase();
  const volume = item.width * item.height * item.depth;
  const prior = WEIGHT_PRIORS.find((candidate) => candidate.terms.some((term) => label.includes(term)));
  if (prior) {
    return {
      kg: Math.round(clamp(prior.base + volume * prior.density, prior.min, prior.max) * 10) / 10,
      basis: prior.basis + " + reconstructed volume",
    };
  }
  return {
    kg: Math.round(clamp(1 + volume * 72, 0.5, 95) * 10) / 10,
    basis: "generic household item + reconstructed volume",
  };
}

function convexHull(points: Array<[number, number]>) {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const unique: Array<[number, number]> = [];
  for (const point of sorted) {
    const previous = unique[unique.length - 1];
    if (!previous || Math.abs(point[0] - previous[0]) > EPSILON || Math.abs(point[1] - previous[1]) > EPSILON) {
      unique.push(point);
    }
  }
  if (unique.length <= 2) return unique;
  const cross = (origin: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]);
  const lower: Array<[number, number]> = [];
  for (const point of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper: Array<[number, number]> = [];
  for (let index = unique.length - 1; index >= 0; index -= 1) {
    const point = unique[index];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function measureYUpBox(scene: THREE.Object3D): RawMeasurement {
  scene.updateMatrixWorld(true);
  const point = new THREE.Vector3();
  const projected: Array<[number, number]> = [];
  let minY = Infinity;
  let maxY = -Infinity;
  scene.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const positions = child.geometry.getAttribute("position");
    if (!positions) return;
    for (let index = 0; index < positions.count; index += 1) {
      point.fromBufferAttribute(positions, index).applyMatrix4(child.matrixWorld);
      projected.push([point.x, point.z]);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }
  });
  const hull = convexHull(projected);
  if (hull.length < 2 || !Number.isFinite(minY) || !Number.isFinite(maxY)) {
    const size = new THREE.Box3().setFromObject(scene).getSize(new THREE.Vector3());
    return { width: Math.max(size.x, 0.001), height: Math.max(size.y, 0.001), depth: Math.max(size.z, 0.001), yaw: 0 };
  }
  let best = { area: Infinity, width: 0, depth: 0, yaw: 0 };
  for (let edge = 0; edge < hull.length; edge += 1) {
    const current = hull[edge];
    const next = hull[(edge + 1) % hull.length];
    const dx = next[0] - current[0];
    const dz = next[1] - current[1];
    const length = Math.hypot(dx, dz);
    if (length < EPSILON) continue;
    const cosine = dx / length;
    const sine = dz / length;
    let minU = Infinity;
    let maxU = -Infinity;
    let minV = Infinity;
    let maxV = -Infinity;
    for (const [x, z] of hull) {
      const u = x * cosine + z * sine;
      const v = -x * sine + z * cosine;
      minU = Math.min(minU, u);
      maxU = Math.max(maxU, u);
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
    const width = maxU - minU;
    const depth = maxV - minV;
    const area = width * depth;
    if (area < best.area) best = { area, width, depth, yaw: Math.atan2(sine, cosine) };
  }
  return { width: Math.max(best.width, 0.001), height: Math.max(maxY - minY, 0.001), depth: Math.max(best.depth, 0.001), yaw: best.yaw };
}

function loadMeasurement(url: string) {
  const cached = measurementCache.get(url);
  if (cached) return cached;
  const request = new GLTFLoader().loadAsync(url).then((gltf) => measureYUpBox(gltf.scene));
  measurementCache.set(url, request);
  return request;
}

function useMeasuredItems(items: PackingItem[]) {
  const [measurements, setMeasurements] = useState<Record<string, RawMeasurement | null>>({});
  useEffect(() => {
    let cancelled = false;
    Promise.all(items.map(async (item) => {
      try {
        return [item.id, await loadMeasurement(item.url)] as const;
      } catch {
        return [item.id, null] as const;
      }
    })).then((entries) => {
      if (!cancelled) setMeasurements(Object.fromEntries(entries) as Record<string, RawMeasurement | null>);
    });
    return () => { cancelled = true; };
  }, [items]);
  const measuredItems = useMemo<MeasuredItem[]>(() => items.map((item) => {
    const measurement = measurements[item.id];
    if (!measurement) return { ...item, boxYaw: 0, meshScale: 1 };
    const scale = THREE.MathUtils.clamp(item.height / Math.max(measurement.height, 0.001), 0.02, 50);
    return {
      ...item,
      width: Math.max(0.03, measurement.width * scale),
      height: Math.max(0.03, measurement.height * scale),
      depth: Math.max(0.03, measurement.depth * scale),
      boxYaw: measurement.yaw,
      meshScale: scale,
    };
  }), [items, measurements]);
  return { measuredItems, measuring: Object.keys(measurements).length < items.length };
}

function rotations(item: MeasuredItem) {
  return [
    { width: item.width, height: item.height, depth: item.depth, rotationY: 0 },
    { width: item.depth, height: item.height, depth: item.width, rotationY: Math.PI / 2 },
  ];
}

function inside(truck: TruckModel, position: [number, number, number], size: Dimensions) {
  const [x, y, z] = position;
  return x >= -EPSILON && y >= -EPSILON && z >= -EPSILON &&
    x + size.width <= truck.dimensions.width + EPSILON &&
    y + size.height <= truck.dimensions.height + EPSILON &&
    z + size.depth <= truck.dimensions.depth + EPSILON;
}

function intersects(a: PlacedItem, position: [number, number, number], size: Dimensions) {
  const [x, y, z] = position;
  const [ax, ay, az] = a.position;
  return x < ax + a.packedWidth - EPSILON && x + size.width > ax + EPSILON &&
    y < ay + a.packedHeight - EPSILON && y + size.height > ay + EPSILON &&
    z < az + a.packedDepth - EPSILON && z + size.depth > az + EPSILON;
}

function overlap(startA: number, lengthA: number, startB: number, lengthB: number) {
  return Math.max(0, Math.min(startA + lengthA, startB + lengthB) - Math.max(startA, startB));
}

function contactScore(truck: TruckModel, placed: PlacedItem[], position: [number, number, number], size: Dimensions) {
  const [x, y, z] = position;
  const { width, depth } = truck.dimensions;
  let score = Math.abs(y) < EPSILON ? size.width * size.depth * 2 : -y * 0.02;
  if (Math.abs(x) < EPSILON || Math.abs(x + size.width - width) < EPSILON) score += size.height * size.depth;
  if (Math.abs(z) < EPSILON || Math.abs(z + size.depth - depth) < EPSILON) score += size.width * size.height;
  for (const item of placed) {
    const [ix, iy, iz] = item.position;
    if (Math.abs(y - (iy + item.packedHeight)) < EPSILON) score += overlap(x, size.width, ix, item.packedWidth) * overlap(z, size.depth, iz, item.packedDepth);
    if (Math.abs(x - (ix + item.packedWidth)) < EPSILON || Math.abs(x + size.width - ix) < EPSILON) score += overlap(y, size.height, iy, item.packedHeight) * overlap(z, size.depth, iz, item.packedDepth);
    if (Math.abs(z - (iz + item.packedDepth)) < EPSILON || Math.abs(z + size.depth - iz) < EPSILON) score += overlap(x, size.width, ix, item.packedWidth) * overlap(y, size.height, iy, item.packedHeight);
  }
  return score;
}

function candidates(truck: TruckModel, placed: PlacedItem[], size: Dimensions) {
  const { width, height, depth } = truck.dimensions;
  const xs = new Set<number>([0, width - size.width]);
  const ys = new Set<number>([0, height - size.height]);
  const zs = new Set<number>([0, depth - size.depth]);
  for (const item of placed) {
    xs.add(item.position[0]);
    xs.add(item.position[0] + item.packedWidth);
    xs.add(item.position[0] - size.width);
    xs.add(item.position[0] + item.packedWidth - size.width);
    ys.add(item.position[1]);
    ys.add(item.position[1] + item.packedHeight);
    zs.add(item.position[2]);
    zs.add(item.position[2] + item.packedDepth);
    zs.add(item.position[2] - size.depth);
    zs.add(item.position[2] + item.packedDepth - size.depth);
  }
  const unique = new Map<string, [number, number, number]>();
  for (const x of xs) for (const y of ys) for (const z of zs) {
    const position: [number, number, number] = [x, y, z];
    if (!inside(truck, position, size)) continue;
    unique.set(position.map((value) => value.toFixed(5)).join(":"), position);
  }
  return [...unique.values()];
}

function packOrder(truck: TruckModel, items: MeasuredItem[]) {
  const packed: PlacedItem[] = [];
  const unfit: MeasuredItem[] = [];
  for (const item of items) {
    let best: { score: number; position: [number, number, number]; size: Dimensions; rotationY: number } | null = null;
    for (const orientation of rotations(item)) {
      for (const position of candidates(truck, packed, orientation)) {
        if (packed.some((placed) => intersects(placed, position, orientation))) continue;
        const score = contactScore(truck, packed, position, orientation);
        if (!best || score > best.score) best = { score, position, size: orientation, rotationY: orientation.rotationY };
      }
    }
    if (!best) {
      unfit.push(item);
      continue;
    }
    packed.push({
      ...item,
      packedWidth: best.size.width,
      packedHeight: best.size.height,
      packedDepth: best.size.depth,
      packedRotationY: best.rotationY,
      position: best.position,
    });
  }
  const volume = packed.reduce((sum, item) => sum + item.width * item.height * item.depth, 0);
  const boundingVolume = truck.dimensions.width * truck.dimensions.height * truck.dimensions.depth;
  return { packed, unfit, utilization: volume / boundingVolume };
}

function packItems(truck: TruckModel, items: MeasuredItem[], strategy: PackingStrategy, weights: Record<string, number>) {
  const orders = strategy === "weight" ? [
    [...items].sort((a, b) => (weights[b.id] ?? 0) - (weights[a.id] ?? 0)),
  ] : strategy === "footprint" ? [
    [...items].sort((a, b) => b.width * b.depth - a.width * a.depth),
  ] : [
    [...items].sort((a, b) => b.width * b.height * b.depth - a.width * a.height * a.depth),
    [...items].sort((a, b) => b.height - a.height),
    [...items].sort((a, b) => b.width - a.width),
    [...items].sort((a, b) => b.depth - a.depth),
    [...items].sort((a, b) => b.width * b.depth - a.width * a.depth),
  ];
  return orders
    .map((order) => packOrder(truck, order))
    .sort((a, b) => b.packed.length - a.packed.length || b.utilization - a.utilization)[0] ?? { packed: [], unfit: items, utilization: 0 };
}


function optimizedTripCount(truck: TruckModel, items: MeasuredItem[], weights: Record<string, number>) {
  let remaining = items.filter((item) => rotations(item).some((orientation) => orientation.width <= truck.dimensions.width + EPSILON && orientation.height <= truck.dimensions.height + EPSILON && orientation.depth <= truck.dimensions.depth + EPSILON) && (weights[item.id] ?? 0) <= truck.cargo.payloadKg);
  let trips = 0;
  while (remaining.length > 0 && trips <= items.length) {
    const proposal = packItems(truck, remaining, "balanced", weights).packed;
    const chosen: MeasuredItem[] = [];
    let payload = 0;
    for (const item of proposal) {
      const weight = weights[item.id] ?? 0;
      if (payload + weight <= truck.cargo.payloadKg + EPSILON) { chosen.push(item); payload += weight; }
    }
    if (chosen.length === 0) break;
    const chosenIds = new Set(chosen.map((item) => item.id));
    remaining = remaining.filter((item) => !chosenIds.has(item.id));
    trips += 1;
  }
  return trips;
}

function VehicleModel({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const model = useMemo(() => {
    const clone = scene.clone(true);
    const meshes: THREE.Mesh[] = [];
    clone.traverse((child) => { if (child instanceof THREE.Mesh) meshes.push(child); });
    for (const mesh of meshes) {
      const sources = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      mesh.material = sources.map((source) => {
        const original = source as THREE.MeshStandardMaterial;
        return new THREE.MeshBasicMaterial({
          color: original.color?.clone() ?? new THREE.Color("#9fb4ac"),
          map: original.map ?? null,
          vertexColors: original.vertexColors,
          transparent: true,
          opacity: 0.22,
          depthWrite: false,
          side: THREE.DoubleSide,
          toneMapped: false,
        });
      });
      const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry, 24),
        new THREE.LineBasicMaterial({ color: "#30483f", transparent: true, opacity: 0.82 }),
      );
      outline.renderOrder = 2;
      mesh.add(outline);
      mesh.renderOrder = -2;
    }
    return clone;
  }, [scene]);
  return <primitive object={model} />;
}

function CargoEnvelope({ truck }: { truck: TruckModel }) {
  const [cargoX, cargoFloor, cargoZ] = truck.cargoOffset;
  return (
    <mesh position={[cargoX, cargoFloor + truck.dimensions.height / 2, cargoZ]}>
      <boxGeometry args={[truck.dimensions.width, truck.dimensions.height, truck.dimensions.depth]} />
      <meshBasicMaterial color="#79d9b5" transparent opacity={0.045} depthWrite={false} side={THREE.DoubleSide} />
      <Edges color="#238263" lineWidth={2.1} />
    </mesh>
  );
}

function brightMaterial(source: THREE.Material, hasVertexColors: boolean) {
  const original = source as THREE.MeshStandardMaterial;
  return new THREE.MeshBasicMaterial({
    color: original.color?.clone() ?? new THREE.Color("#ffffff"),
    map: original.map ?? null,
    vertexColors: hasVertexColors || original.vertexColors,
    transparent: original.transparent,
    opacity: original.opacity,
    alphaTest: original.alphaTest,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
}

function initialTransform(item: PlacedItem, truck: TruckModel): ItemTransform {
  const [cargoX, cargoFloor, cargoZ] = truck.cargoOffset;
  const [x, y, z] = item.position;
  return {
    position: [
      cargoX + x + item.packedWidth / 2 - truck.dimensions.width / 2,
      cargoFloor + y + item.packedHeight / 2,
      cargoZ + z + item.packedDepth / 2 - truck.dimensions.depth / 2,
    ],
    rotationY: item.packedRotationY,
  };
}

function PackedItemMesh({ item, truck, active, mode, transform, onSelect, onTransform }: {
  item: PlacedItem;
  truck: TruckModel;
  active: boolean;
  mode: EditMode;
  transform?: ItemTransform;
  onSelect: (id: string) => void;
  onTransform: (id: string, transform: ItemTransform) => void;
}) {
  const { scene } = useGLTF(item.url);
  const groupRef = useRef<THREE.Group>(null!);
  const current = transform ?? initialTransform(item, truck);
  const prepared = useMemo(() => {
    const model = scene.clone(true);
    model.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      const hasVertexColors = Boolean(child.geometry.getAttribute("color"));
      child.material = Array.isArray(child.material)
        ? child.material.map((material) => brightMaterial(material, hasVertexColors))
        : brightMaterial(child.material, hasVertexColors);
      child.renderOrder = 1;
    });
    const aligned = new THREE.Group();
    aligned.add(model);
    aligned.rotation.y = item.boxYaw;
    aligned.scale.setScalar(item.meshScale);
    aligned.updateMatrixWorld(true);
    const center = new THREE.Box3().setFromObject(aligned).getCenter(new THREE.Vector3());
    aligned.position.set(-center.x, -center.y, -center.z);
    return aligned;
  }, [scene, item.boxYaw, item.meshScale]);
  const select = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    onSelect(item.id);
  };
  const saveTransform = () => {
    const group = groupRef.current;
    if (!group) return;
    onTransform(item.id, { position: [group.position.x, group.position.y, group.position.z], rotationY: group.rotation.y });
  };
  return (
    <>
      <group ref={groupRef} position={current.position} rotation={[0, current.rotationY, 0]} onClick={select} onPointerDown={select}>
        <primitive object={prepared} />
        <mesh>
          <boxGeometry args={[item.width, item.height, item.depth]} />
          <meshBasicMaterial color={item.color} transparent opacity={active ? 0.075 : 0.025} depthWrite={false} />
          <Edges color={active ? "#e56f2e" : item.color} lineWidth={active ? 2.8 : 1.6} />
        </mesh>
      </group>
      {active && (
        <TransformControls
          object={groupRef}
          mode={mode}
          space={mode === "translate" ? "world" : "local"}
          showX={mode === "translate"}
          showY
          showZ={mode === "translate"}
          translationSnap={0.01}
          rotationSnap={Math.PI / 36}
          size={0.72}
          onMouseUp={saveTransform}
        />
      )}
    </>
  );
}

function CameraRig({ truck, view }: { truck: TruckModel; view: CameraView }) {
  const { camera } = useThree();
  const controls = useRef<any>(null);
  useEffect(() => {
    const [, cargoFloor, cargoZ] = truck.cargoOffset;
    const targetY = cargoFloor + truck.dimensions.height * 0.45;
    camera.up.set(0, 1, 0);
    if (view === "top") {
      camera.up.set(0, 0, -1);
      camera.position.set(0, 5.2, cargoZ);
    } else if (view === "side") {
      camera.position.set(4.6, targetY + 0.15, cargoZ);
    } else {
      camera.position.set(3.7, 2.55, 4.05);
    }
    camera.lookAt(0, targetY, cargoZ);
    controls.current?.target.set(0, targetY, cargoZ);
    controls.current?.update();
  }, [camera, truck, view]);
  return <OrbitControls ref={controls} makeDefault enablePan={false} minDistance={2} maxDistance={10} />;
}

function PackingScene({ truck, packed, activeId, mode, view, transforms, onSelect, onTransform }: {
  truck: TruckModel;
  packed: PlacedItem[];
  activeId: string | null;
  mode: EditMode;
  view: CameraView;
  transforms: Record<string, ItemTransform>;
  onSelect: (id: string | null) => void;
  onTransform: (id: string, transform: ItemTransform) => void;
}) {
  return (
    <Canvas camera={{ position: [3.7, 2.55, 4.05], fov: 42 }} dpr={[1, 1.75]} onPointerMissed={() => onSelect(null)}>
      <color attach="background" args={["#f7faf9"]} />
      <ambientLight intensity={2.2} />
      <hemisphereLight args={["#ffffff", "#dbe7e2", 1.75]} />
      <directionalLight position={[5, 9, 7]} intensity={2.1} />
      <CameraRig truck={truck} view={view} />
      <gridHelper args={[10, 20, "#c5d5cf", "#e4ece9"]} position={[0, -0.008, 0]} />
      <Suspense fallback={null}>
        <VehicleModel url={truck.modelUrl} />
        <CargoEnvelope truck={truck} />
        {packed.map((item) => (
          <PackedItemMesh
            key={item.id}
            item={item}
            truck={truck}
            active={activeId === item.id}
            mode={mode}
            transform={transforms[item.id]}
            onSelect={(id) => onSelect(id)}
            onTransform={onTransform}
          />
        ))}
      </Suspense>
    </Canvas>
  );
}

function ThumbnailCamera({ targetY = 0.55 }: { targetY?: number }) {
  const { camera } = useThree();
  useEffect(() => {
    camera.lookAt(0, targetY, 0);
    camera.updateProjectionMatrix();
  }, [camera, targetY]);
  return null;
}

function VehicleThumbnail({ truck }: { truck: TruckModel }) {
  return (
    <Canvas camera={{ position: [2.7, 1.8, 3.2], fov: 37 }} dpr={[1, 1.4]}>
      <color attach="background" args={["#f1f6f4"]} />
      <ThumbnailCamera />
      <Suspense fallback={null}><VehicleModel url={truck.modelUrl} /></Suspense>
    </Canvas>
  );
}

const itemThumbnailCache = new Map<string, Promise<string>>();
let itemThumbnailRenderer: THREE.WebGLRenderer | null = null;

function renderItemThumbnail(item: MeasuredItem) {
  const cached = itemThumbnailCache.get(item.url);
  if (cached) return cached;
  const pending = new Promise<string>((resolve, reject) => {
    new GLTFLoader().load(item.url, (gltf) => {
      const root = gltf.scene.clone(true);
      root.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        const hasVertexColors = Boolean(child.geometry.getAttribute("color"));
        child.material = Array.isArray(child.material)
          ? child.material.map((material) => brightMaterial(material, hasVertexColors))
          : brightMaterial(child.material, hasVertexColors);
      });
      root.rotation.y = item.boxYaw;
      root.updateMatrixWorld(true);
      const bounds = new THREE.Box3().setFromObject(root);
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      root.position.set(-center.x, -center.y, -center.z);
      const preview = new THREE.Scene();
      preview.background = new THREE.Color("#f0f6f3");
      const group = new THREE.Group();
      group.scale.setScalar(1.45 / Math.max(size.x, size.y, size.z, 0.001));
      group.add(root);
      preview.add(group);
      const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 20);
      camera.position.set(1.8, 1.25, 2.2);
      camera.lookAt(0, 0, 0);
      if (!itemThumbnailRenderer) {
        itemThumbnailRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
        itemThumbnailRenderer.outputColorSpace = THREE.SRGBColorSpace;
      }
      itemThumbnailRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      itemThumbnailRenderer.setSize(84, 84, false);
      itemThumbnailRenderer.render(preview, camera);
      resolve(itemThumbnailRenderer.domElement.toDataURL("image/png"));
    }, undefined, reject);
  });
  itemThumbnailCache.set(item.url, pending);
  return pending;
}

function ItemThumbnail({ item }: { item: MeasuredItem }) {
  const [source, setSource] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    renderItemThumbnail(item).then((url) => { if (active) setSource(url); }).catch(() => { if (active) setSource(null); });
    return () => { active = false; };
  }, [item]);
  return source ? <img src={source} alt={item.name + " preview"} /> : <span className="item-thumb-loading" aria-hidden="true" />;
}


export function PackingItemThumbnail({ item }: { item: PackingItem }) {
  return <ItemThumbnail item={{ ...item, boxYaw: 0, meshScale: 1 }} />;
}

export function estimatePackingItemWeight(item: PackingItem) {
  return estimateWeight({ ...item, boxYaw: 0, meshScale: 1 });
}

function itemsOverlap(packed: PlacedItem[], transforms: Record<string, ItemTransform>, truck: TruckModel) {
  const boxes = packed.map((item) => {
    const transform = transforms[item.id] ?? initialTransform(item, truck);
    const cosine = Math.abs(Math.cos(transform.rotationY));
    const sine = Math.abs(Math.sin(transform.rotationY));
    const halfX = (cosine * item.width + sine * item.depth) / 2;
    const halfZ = (sine * item.width + cosine * item.depth) / 2;
    return {
      minX: transform.position[0] - halfX,
      maxX: transform.position[0] + halfX,
      minY: transform.position[1] - item.height / 2,
      maxY: transform.position[1] + item.height / 2,
      minZ: transform.position[2] - halfZ,
      maxZ: transform.position[2] + halfZ,
    };
  });
  for (let first = 0; first < boxes.length; first += 1) {
    for (let second = first + 1; second < boxes.length; second += 1) {
      const a = boxes[first];
      const b = boxes[second];
      if (a.minX < b.maxX - 0.004 && a.maxX > b.minX + 0.004 &&
        a.minY < b.maxY - 0.004 && a.maxY > b.minY + 0.004 &&
        a.minZ < b.maxZ - 0.004 && a.maxZ > b.minZ + 0.004) return true;
    }
  }
  return false;
}

function MetricRing({ progress, value, label, detail, danger = false }: {
  progress: number;
  value: string;
  label: string;
  detail: string;
  danger?: boolean;
}) {
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - clamp(progress, 0, 1));
  return (
    <div className={danger ? "truck-metric danger" : "truck-metric"}>
      <div className="truck-metric-ring">
        <svg viewBox="0 0 80 80" aria-hidden="true">
          <circle className="metric-track" cx="40" cy="40" r={radius} />
          <circle className="metric-value" cx="40" cy="40" r={radius} strokeDasharray={circumference} strokeDashoffset={offset} />
        </svg>
        <strong>{value}</strong>
      </div>
      <span>{label}</span>
      <small>{detail}</small>
    </div>
  );
}

function loadGoogleMaps() {
  if (typeof window === "undefined") return Promise.reject(new Error("Google Maps is only available in the browser"));
  if (window.google?.maps?.places) return Promise.resolve(window.google);
  if (window.__honkpackGoogleMapsPromise) return window.__honkpackGoogleMapsPromise;
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY?.trim();
  if (!apiKey) return Promise.reject(new Error("Google Maps API key is not configured"));
  window.__honkpackGoogleMapsPromise = new Promise((resolve, reject) => {
    window.__honkpackGoogleMapsReady = () => window.google?.maps ? resolve(window.google) : reject(new Error("Google Maps did not initialize"));
    window.gm_authFailure = () => reject(new Error("Google Maps rejected this API key or website domain"));
    const script = document.createElement("script");
    script.id = "honkpack-google-maps";
    script.async = true;
    script.defer = true;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&v=weekly&loading=async&callback=__honkpackGoogleMapsReady`;
    script.onerror = () => reject(new Error("Unable to load Google Maps"));
    document.head.appendChild(script);
  });
  return window.__honkpackGoogleMapsPromise;
}

function GooglePlaceInput({ label, placeholder, value, onChange }: {
  label: string;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
}) {
  const [suggestions, setSuggestions] = useState<Array<{ text: string; placeId?: string }>>([]);
  const [open, setOpen] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  useEffect(() => {
    if (value.trim().length < 2) { setSuggestions([]); return; }
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/maps/autocomplete?q=${encodeURIComponent(value.trim())}`, { signal: controller.signal });
        const payload = await response.json() as { suggestions?: Array<{ text: string; placeId?: string }>; detail?: string };
        if (!response.ok) throw new Error(payload.detail || "Place search unavailable");
        setSuggestions(payload.suggestions ?? []);
        setSearchError(null);
      } catch (error) {
        if (!controller.signal.aborted) setSearchError(error instanceof Error ? error.message : "Place search unavailable");
      }
    }, 220);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [value]);
  return (
    <div className="google-place-field">
      <label>
        <span>{label}</span>
        <input value={value} onChange={(event) => { onChange(event.target.value); setOpen(true); }} onFocus={() => setOpen(true)} onBlur={() => window.setTimeout(() => setOpen(false), 120)} placeholder={placeholder} autoComplete="off" />
      </label>
      {open && suggestions.length > 0 && (
        <div className="google-place-suggestions" role="listbox" aria-label={label + " suggestions"}>
          {suggestions.map((suggestion) => (
            <button key={(suggestion.placeId || "") + suggestion.text} type="button" role="option" onMouseDown={(event) => event.preventDefault()} onClick={() => { onChange(suggestion.text); setOpen(false); }}>
              {suggestion.text}
            </button>
          ))}
        </div>
      )}
      {searchError && <small className="google-place-error">{searchError}</small>}
    </div>
  );
}

function decodeRoutePolyline(encoded: string) {
  const points: Array<{ lat: number; lng: number }> = [];
  let index = 0;
  let latitude = 0;
  let longitude = 0;
  while (index < encoded.length) {
    let result = 0;
    let shift = 0;
    let byte = 0;
    do {
      byte = encoded.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20 && index < encoded.length);
    latitude += result & 1 ? ~(result >> 1) : result >> 1;
    result = 0;
    shift = 0;
    do {
      byte = encoded.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20 && index < encoded.length);
    longitude += result & 1 ? ~(result >> 1) : result >> 1;
    points.push({ lat: latitude / 1e5, lng: longitude / 1e5 });
  }
  return points;
}

function RouteMap({ encodedPolyline }: { encodedPolyline: string | null }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    loadGoogleMaps().then((google) => {
      if (cancelled || !hostRef.current) return;
      const map = new google.maps.Map(hostRef.current, {
        center: { lat: 42.3601, lng: -71.0942 },
        zoom: 12,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false,
      });
      if (!encodedPolyline) return;
      const path = decodeRoutePolyline(encodedPolyline);
      if (path.length < 2) return;
      new google.maps.Polyline({ path, map, strokeColor: "#26a875", strokeOpacity: 1, strokeWeight: 6 });
      new google.maps.Marker({ position: path[0], map, label: "A" });
      new google.maps.Marker({ position: path[path.length - 1], map, label: "B" });
      const bounds = new google.maps.LatLngBounds();
      path.forEach((point) => bounds.extend(point));
      map.fitBounds(bounds, 38);
    }).catch((error) => {
      if (!cancelled) setMapError(error instanceof Error ? error.message : "Map unavailable");
    });
    return () => { cancelled = true; };
  }, [encodedPolyline]);
  return (
    <div className="route-map route-map-google" aria-label="Google route map">
      <div ref={hostRef} className="route-map-canvas" />
      {mapError && <div className="route-map-error">{mapError}</div>}
    </div>
  );
}

type TripPlan = {
  id: number;
  itemIds: Set<string>;
  truckId: string;
  transforms: Record<string, ItemTransform>;
};


function createMinimumTripPlans(items: PackingItem[], truckId = "cargo-van"): TripPlan[] {
  const truck = TRUCK_MODELS.find((candidate) => candidate.id === truckId) ?? TRUCK_MODELS[2];
  const usableVolume = truck.dimensions.width * truck.dimensions.height * truck.dimensions.depth * 0.82;
  const weighted = items.map((item) => ({
    item,
    volume: item.width * item.height * item.depth,
    weight: estimateWeight({ ...item, boxYaw: 0, meshScale: 1 }).kg,
  }));
  const totalVolume = weighted.reduce((sum, entry) => sum + entry.volume, 0);
  const totalWeight = weighted.reduce((sum, entry) => sum + entry.weight, 0);
  const minimum = Math.max(1, Math.ceil(totalVolume / Math.max(usableVolume, 0.001)), Math.ceil(totalWeight / truck.cargo.payloadKg));
  const buckets = Array.from({ length: minimum }, (_, index) => ({ id: index + 1, itemIds: new Set<string>(), truckId: truck.id, transforms: {}, volume: 0, weight: 0 }));
  for (const entry of [...weighted].sort((a, b) => Math.max(b.volume / usableVolume, b.weight / truck.cargo.payloadKg) - Math.max(a.volume / usableVolume, a.weight / truck.cargo.payloadKg))) {
    const target = buckets.reduce((best, bucket) => {
      const score = Math.max(bucket.volume / usableVolume, bucket.weight / truck.cargo.payloadKg);
      const bestScore = Math.max(best.volume / usableVolume, best.weight / truck.cargo.payloadKg);
      return score < bestScore ? bucket : best;
    });
    target.itemIds.add(entry.item.id);
    target.volume += entry.volume;
    target.weight += entry.weight;
  }
  return buckets.map(({ volume: _volume, weight: _weight, ...trip }) => trip);
}

export function TruckFitting({ items, onViewScene }: { items: PackingItem[]; onViewScene: () => void }) {
  const [tripPlans, setTripPlans] = useState<TripPlan[]>(() => [{ id: 1, itemIds: new Set(items.map((item) => item.id)), truckId: "cargo-van", transforms: {} }]);
  const [activeTripId, setActiveTripId] = useState(1);
  const [mapOpen, setMapOpen] = useState(true);
  const [routeStart, setRouteStart] = useState("");
  const [routeEnd, setRouteEnd] = useState("");
  const [distanceKm, setDistanceKm] = useState<number | null>(null);
  const [routeDurationHours, setRouteDurationHours] = useState<number | null>(null);
  const [routePolyline, setRoutePolyline] = useState<string | null>(null);
  const [routeLoading, setRouteLoading] = useState(false);
  const [routeError, setRouteError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [mode, setMode] = useState<EditMode>("translate");
  const [cameraView, setCameraView] = useState<CameraView>("perspective");
  const [packingStrategy, setPackingStrategy] = useState<PackingStrategy>("balanced");
  const [weightOverrides, setWeightOverrides] = useState<Record<string, number>>({});
  const [editingWeightId, setEditingWeightId] = useState<string | null>(null);
  const [weightDraft, setWeightDraft] = useState("");
  const { measuredItems } = useMeasuredItems(items);
  const activePlan = tripPlans.find((trip) => trip.id === activeTripId) ?? tripPlans[0];
  const selectedIds = activePlan.itemIds;
  const truck = TRUCK_MODELS.find((item) => item.id === activePlan.truckId) ?? TRUCK_MODELS[2];
  const transforms = activePlan.transforms;
  const selectedItems = useMemo(() => measuredItems.filter((item) => selectedIds.has(item.id)), [measuredItems, selectedIds]);
  const estimates = useMemo(() => Object.fromEntries(measuredItems.map((item) => [item.id, estimateWeight(item)])), [measuredItems]);
  const packingWeights = useMemo(() => Object.fromEntries(measuredItems.map((item) => [item.id, weightOverrides[item.id] ?? estimates[item.id]?.kg ?? 0])), [estimates, measuredItems, weightOverrides]);
  const result = useMemo(() => packItems(truck, selectedItems, packingStrategy, packingWeights), [packingStrategy, packingWeights, selectedItems, truck]);
  const packedWeight = result.packed.reduce((sum, item) => sum + (packingWeights[item.id] ?? 0), 0);
  const percentage = Math.round(result.utilization * 100);
  const overloaded = packedWeight > truck.cargo.payloadKg;
  const overlapWarning = useMemo(() => itemsOverlap(result.packed, transforms, truck), [result.packed, transforms, truck]);
  const routeReady = distanceKm !== null && routeDurationHours !== null;
  const roundTripHours = routeDurationHours === null ? 0 : Math.max(1, routeDurationHours * 2);
  const nonEmptyTrips = tripPlans.filter((trip) => trip.itemIds.size > 0);
  const quotes = useMemo(() => TRUCK_MODELS.map((model) => {
    const trips = optimizedTripCount(model, measuredItems, packingWeights);
    return { model, trips, total: trips * roundTripHours * model.cargo.pricePerHour };
  }), [measuredItems, packingWeights, roundTripHours]);
  const lowestPrice = routeReady ? Math.min(...quotes.filter((quote) => quote.trips > 0).map((quote) => quote.total)) : Number.POSITIVE_INFINITY;
  const fewestTrips = Math.min(...quotes.filter((quote) => quote.trips > 0).map((quote) => quote.trips));
  const totalPrice = nonEmptyTrips.length * roundTripHours * truck.cargo.pricePerHour;

  const updateActivePlan = (update: (trip: TripPlan) => TripPlan) => setTripPlans((current) => current.map((trip) => trip.id === activeTripId ? update(trip) : trip));
  const setTransforms = (update: Record<string, ItemTransform> | ((current: Record<string, ItemTransform>) => Record<string, ItemTransform>)) => {
    updateActivePlan((trip) => ({ ...trip, transforms: typeof update === "function" ? update(trip.transforms) : update }));
  };
  const setTruckId = (truckId: string) => updateActivePlan((trip) => ({ ...trip, truckId, transforms: {} }));
  const assignItem = (itemId: string, target: string) => {
    const targetId = target === "unassigned" ? null : Number(target);
    setTripPlans((current) => current.map((trip) => {
      const itemIds = new Set(trip.itemIds);
      itemIds.delete(itemId);
      if (trip.id === targetId) itemIds.add(itemId);
      const transforms = { ...trip.transforms };
      delete transforms[itemId];
      return { ...trip, itemIds, transforms };
    }));
    if (targetId !== null) setActiveTripId(targetId);
  };
  const addTrip = () => {
    const id = Math.max(0, ...tripPlans.map((trip) => trip.id)) + 1;
    setTripPlans((current) => [...current, { id, itemIds: new Set(), truckId: activePlan.truckId, transforms: {} }]);
    setActiveTripId(id);
  };
  const removeTrip = (id: number) => {
    const target = tripPlans.find((trip) => trip.id === id);
    if (!target || target.itemIds.size > 0 || tripPlans.length === 1) return;
    const remaining = tripPlans.filter((trip) => trip.id !== id);
    const desiredActiveId = activeTripId === id ? remaining[0].id : activeTripId;
    const activeIndex = Math.max(0, remaining.findIndex((trip) => trip.id === desiredActiveId));
    setTripPlans(remaining.map((trip, index) => ({ ...trip, id: index + 1 })));
    setActiveTripId(activeIndex + 1);
  };
  const calculateRoute = async () => {
    if (!routeStart.trim() || !routeEnd.trim()) return;
    setRouteLoading(true);
    setRouteError(null);
    try {
      const response = await fetch(`${API_BASE}/api/maps/route`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ origin: routeStart.trim(), destination: routeEnd.trim() }) });
      const payload = await response.json() as { distanceKm?: number; durationSeconds?: number; encodedPolyline?: string; detail?: string };
      if (!response.ok || payload.distanceKm === undefined) throw new Error(payload.detail || "Route unavailable");
      setDistanceKm(payload.distanceKm);
      setRouteDurationHours((payload.durationSeconds ?? 0) / 3600);
      setRoutePolyline(payload.encodedPolyline ?? null);
      setMapOpen(true);
    } catch (error) { setRouteError(error instanceof Error ? error.message : "Route unavailable"); }
    finally { setRouteLoading(false); }
  };
  const updateTransform = (id: string, transform: ItemTransform) => setTransforms((current) => ({ ...current, [id]: transform }));
  const beginWeightEdit = (item: MeasuredItem) => {
    setEditingWeightId(item.id);
    setWeightDraft(String(weightOverrides[item.id] ?? estimates[item.id]?.kg ?? 0));
  };
  const commitWeight = (id: string) => {
    const value = Number(weightDraft);
    if (Number.isFinite(value) && value > 0) setWeightOverrides((current) => ({ ...current, [id]: Math.round(value * 10) / 10 }));
    setEditingWeightId(null);
  };
  useEffect(() => { setActiveId(null); }, [activeTripId]);

  return (
    <div className="truck-fit">
      <aside className="truck-manifest" aria-label="Assign items to trips">
        <div className="truck-manifest-header">
          <div><strong>Items</strong><span>{selectedIds.size} in Trip {activeTripId} · {result.packed.length}/{selectedItems.length} fit</span></div>
        </div>
        <div className="truck-manifest-list">
          {measuredItems.map((item) => {
            const assignedPlan = tripPlans.find((trip) => trip.itemIds.has(item.id));
            const selected = assignedPlan?.id === activeTripId;
            const fits = result.packed.some((packed) => packed.id === item.id);
            const estimate = estimates[item.id] ?? { kg: 0, basis: "item type" };
            const displayedWeight = weightOverrides[item.id] ?? estimate.kg;
            const manual = weightOverrides[item.id] !== undefined;
            const impossibleForVehicle = selected && (!rotations(item).some((orientation) => orientation.width <= truck.dimensions.width + EPSILON && orientation.height <= truck.dimensions.height + EPSILON && orientation.depth <= truck.dimensions.depth + EPSILON) || displayedWeight > truck.cargo.payloadKg);
            const currentlyUnfit = selected && !fits;
            return (
              <div key={item.id} className={["truck-item-row", selected ? "selected" : "not-selected", currentlyUnfit ? "unfit" : "", activeId === item.id ? "active" : ""].join(" ")} onMouseEnter={() => selected && setActiveId(item.id)} onClick={() => selected && setActiveId(item.id)}>
                <span className="item-thumb"><ItemThumbnail item={item} /></span>
                <span className="truck-item-copy">
                  <strong>{item.name}</strong>
                  <small>{item.width.toFixed(2)} × {item.height.toFixed(2)} × {item.depth.toFixed(2)} m</small>
                  {currentlyUnfit && <small className="item-fit-error">{impossibleForVehicle ? "Too large for this vehicle" : "Not packed in this trip"}</small>}
                  {editingWeightId === item.id ? (
                    <span className="weight-editor" onClick={(event) => event.stopPropagation()}>
                      <input autoFocus type="number" min="0.1" step="0.1" value={weightDraft} aria-label={item.name + " weight in kilograms"} onChange={(event) => setWeightDraft(event.target.value)} onBlur={() => commitWeight(item.id)} onKeyDown={(event) => { if (event.key === "Enter") commitWeight(item.id); if (event.key === "Escape") setEditingWeightId(null); }} />
                      <em>kg</em>
                    </span>
                  ) : (
                    <button className="item-weight" title={(manual ? "User-entered weight" : "Estimated from " + estimate.basis) + ". Double-click to edit."} onClick={(event) => event.stopPropagation()} onDoubleClick={(event) => { event.stopPropagation(); beginWeightEdit(item); }}>
                      {manual ? "" : "~"}{displayedWeight.toFixed(1)} kg
                    </button>
                  )}
                </span>
                <select className="item-trip-select" value={assignedPlan?.id ?? "unassigned"} onClick={(event) => event.stopPropagation()} onChange={(event) => assignItem(item.id, event.target.value)} aria-label={"Assign " + item.name}>
                  <option value="unassigned">Unassigned</option>
                  {tripPlans.map((trip) => <option key={trip.id} value={trip.id}>Trip {trip.id}</option>)}
                </select>
              </div>
            );
          })}
        </div>
      </aside>

      <section className="truck-fit-stage">
        <div className="trip-planner-bar">
          <button className={mapOpen ? "map-trigger active" : "map-trigger"} onClick={() => routeReady && setMapOpen((open) => !open)} disabled={!routeReady}>Map</button>
          <div className="trip-tabs" role="tablist" aria-label="Trips">
            {tripPlans.map((trip) => (
              <span key={trip.id} className={"trip-tab " + (activeTripId === trip.id ? "active" : "")}>
                <button onClick={() => setActiveTripId(trip.id)}>Trip {trip.id}</button>
                {trip.itemIds.size === 0 && tripPlans.length > 1 && <button className="remove-trip" onClick={() => removeTrip(trip.id)} aria-label={"Remove Trip " + trip.id}>×</button>}
              </span>
            ))}
            <button className="add-trip" onClick={addTrip} aria-label="Add trip">+</button>
          </div>
          <div className="trip-total"><strong>{nonEmptyTrips.length} {nonEmptyTrips.length === 1 ? "round trip" : "round trips"}</strong><span>{routeReady ? "$" + Math.round(totalPrice) + " total" : "Route required"}</span></div>
        </div>
        {!routeReady && <div className="route-required-backdrop" />}
        {(mapOpen || !routeReady) && (
          <div className={"route-panel route-panel-with-map " + (!routeReady ? "route-required" : "")}>
            {!routeReady && <div className="route-required-copy"><strong>Choose your route first</strong><span>Distance and travel time are required before vehicle prices can be calculated.</span></div>}
            <GooglePlaceInput label="From" value={routeStart} onChange={setRouteStart} placeholder="Pickup address" />
            <GooglePlaceInput label="To" value={routeEnd} onChange={setRouteEnd} placeholder="Destination address" />
            <button onClick={calculateRoute} disabled={routeLoading || !routeStart.trim() || !routeEnd.trim()}>{routeLoading ? "Calculating…" : "Calculate"}</button>
            {distanceKm !== null && <strong>{distanceKm.toFixed(1)} km · {roundTripHours.toFixed(1)} hr round trip</strong>}
            {routeError && <small>{routeError}</small>}
            <RouteMap encodedPolyline={routePolyline} />
          </div>
        )}
        <div className="result-switch truck-fit-view-switch">
          <button onClick={onViewScene}>Back to the scene</button>
        </div>
        <div className="truck-edit-tools truck-edit-float" aria-label="Packing and view controls">
          <button className={mode === "translate" ? "active" : ""} onClick={() => setMode("translate")}>Move</button>
          <button className={mode === "rotate" ? "active" : ""} onClick={() => setMode("rotate")}>Rotate</button>
          <span className="tool-divider" />
          <button className={cameraView === "perspective" ? "active" : ""} onClick={() => setCameraView("perspective")}>3D</button>
          <button className={cameraView === "top" ? "active" : ""} onClick={() => setCameraView("top")}>Top</button>
          <button className={cameraView === "side" ? "active" : ""} onClick={() => setCameraView("side")}>Side</button>
          <span className="tool-divider" />
          <div className="auto-pack-control">
            <button onClick={() => { setPackingStrategy("balanced"); setTransforms({}); }}>Auto-pack</button>
            <div className="auto-pack-menu" role="menu">
              <button role="menuitem" onClick={() => { setPackingStrategy("weight"); setTransforms({}); }}><strong>Heavy items first</strong><small>Pack weight from the floor up</small></button>
              <button role="menuitem" onClick={() => { setPackingStrategy("footprint"); setTransforms({}); }}><strong>Largest base first</strong><small>Prioritize floor footprint</small></button>
            </div>
          </div>
        </div>
        <div className="truck-fit-canvas">
          <PackingScene truck={truck} packed={result.packed} activeId={activeId} mode={mode} view={cameraView} transforms={transforms} onSelect={setActiveId} onTransform={updateTransform} />
        </div>
      </section>

      <aside className="truck-dashboard" aria-label="Packing summary and vehicle comparison">
        <div className="truck-dashboard-heading"><strong>Packing summary</strong></div>
        {overlapWarning && <div className="packing-overlap-warning">Items overlap — move them apart or run Auto-pack.</div>}
        <div className="truck-metrics">
          <MetricRing progress={result.utilization} value={percentage + "%"} label="Volume" detail="Used" />
          <MetricRing progress={packedWeight / truck.cargo.payloadKg} value={Math.round(packedWeight) + "kg"} label="Payload" detail={"of " + truck.cargo.payloadKg + " kg"} danger={overloaded} />
        </div>
        <div className="vehicle-picker-heading"><strong>Vehicle</strong><span>Compare total move</span></div>
        <div className="vehicle-list">
          {quotes.map(({ model, trips, total }) => {
            const cheapest = routeReady && trips > 0 && Math.abs(total - lowestPrice) < 0.01;
            const fewest = trips === fewestTrips;
            const badge = cheapest && fewest ? "Best price · Fewest trips" : cheapest ? "Lowest price" : fewest ? "Fewest trips" : null;
            return (
              <button key={model.id} className={[activePlan.truckId === model.id ? "selected" : "", cheapest && fewest ? "best-both" : cheapest ? "best-price" : fewest ? "best-trips" : ""].join(" ")} onClick={() => setTruckId(model.id)} aria-pressed={activePlan.truckId === model.id}>
                <span className="vehicle-thumb"><VehicleThumbnail truck={model} /></span>
                <span className="vehicle-copy">
                  <strong>{model.name}</strong>
                  <small>{"$" + model.cargo.pricePerHour + "/hr · " + trips + (trips === 1 ? " round trip" : " round trips")}</small>
                  <em>{routeReady ? "$" + Math.round(total) + " minimum" : "Route required"}</em>
                  {badge && <b>{badge}</b>}
                </span>
                <i aria-hidden="true" />
              </button>
            );
          })}
        </div>
      </aside>
    </div>
  );
}
