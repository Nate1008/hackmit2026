"use client";

import { Edges, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

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
type TruckModel = {
  id: string;
  name: string;
  dimensions: Dimensions;
  modelUrl: string;
  cargoOffset: [number, number, number];
};
type PlacedItem = PackingItem & {
  packedWidth: number;
  packedHeight: number;
  packedDepth: number;
  position: [number, number, number];
};

export const TRUCK_MODELS: TruckModel[] = [
  {
    id: "suv",
    name: "Compact SUV",
    dimensions: { width: 1.12, height: 0.72, depth: 1.15 },
    modelUrl: "/vehicles/suv.glb",
    cargoOffset: [0, 0.42, -0.61],
  },
  {
    id: "pickup",
    name: "Pickup truck",
    dimensions: { width: 1.25, height: 0.48, depth: 1.35 },
    modelUrl: "/vehicles/truck.glb",
    cargoOffset: [0, 0.56, -0.67],
  },
  {
    id: "cargo-van",
    name: "Cargo van",
    dimensions: { width: 1.28, height: 1.02, depth: 1.58 },
    modelUrl: "/vehicles/van.glb",
    cargoOffset: [0, 0.27, -0.57],
  },
  {
    id: "box-truck",
    name: "Box truck",
    dimensions: { width: 1.32, height: 1.22, depth: 1.78 },
    modelUrl: "/vehicles/delivery.glb",
    cargoOffset: [0, 0.39, -0.69],
  },
];

TRUCK_MODELS.forEach((truck) => useGLTF.preload(truck.modelUrl));

const EPSILON = 1e-7;

function rotations(item: PackingItem): Dimensions[] {
  const values = [
    [item.width, item.height, item.depth],
    [item.width, item.depth, item.height],
    [item.height, item.width, item.depth],
    [item.height, item.depth, item.width],
    [item.depth, item.width, item.height],
    [item.depth, item.height, item.width],
  ];
  const seen = new Set<string>();
  return values.flatMap(([width, height, depth]) => {
    const key = `${width.toFixed(5)}:${height.toFixed(5)}:${depth.toFixed(5)}`;
    if (seen.has(key)) return [];
    seen.add(key);
    return [{ width, height, depth }];
  });
}

function inside(truck: Dimensions, position: [number, number, number], size: Dimensions) {
  const [x, y, z] = position;
  return x >= -EPSILON && y >= -EPSILON && z >= -EPSILON &&
    x + size.width <= truck.width + EPSILON &&
    y + size.height <= truck.height + EPSILON &&
    z + size.depth <= truck.depth + EPSILON;
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

function contactScore(truck: Dimensions, placed: PlacedItem[], position: [number, number, number], size: Dimensions) {
  const [x, y, z] = position;
  let score = Math.abs(y) < EPSILON ? size.width * size.depth * 2 : -y * 0.02;
  if (Math.abs(x) < EPSILON || Math.abs(x + size.width - truck.width) < EPSILON) score += size.height * size.depth;
  if (Math.abs(z) < EPSILON || Math.abs(z + size.depth - truck.depth) < EPSILON) score += size.width * size.height;
  for (const item of placed) {
    const [ix, iy, iz] = item.position;
    if (Math.abs(y - (iy + item.packedHeight)) < EPSILON) {
      score += overlap(x, size.width, ix, item.packedWidth) * overlap(z, size.depth, iz, item.packedDepth);
    }
    if (Math.abs(x - (ix + item.packedWidth)) < EPSILON || Math.abs(x + size.width - ix) < EPSILON) {
      score += overlap(y, size.height, iy, item.packedHeight) * overlap(z, size.depth, iz, item.packedDepth);
    }
    if (Math.abs(z - (iz + item.packedDepth)) < EPSILON || Math.abs(z + size.depth - iz) < EPSILON) {
      score += overlap(x, size.width, ix, item.packedWidth) * overlap(y, size.height, iy, item.packedHeight);
    }
  }
  return score;
}

function candidates(truck: Dimensions, placed: PlacedItem[], size: Dimensions) {
  const points: [number, number, number][] = [
    [0, 0, 0],
    [truck.width - size.width, 0, 0],
    [0, 0, truck.depth - size.depth],
    [truck.width - size.width, 0, truck.depth - size.depth],
  ];
  for (const item of placed) {
    const [x, y, z] = item.position;
    const xs = [x, x + item.packedWidth, x - size.width, x + item.packedWidth - size.width];
    const ys = [0, y, y + item.packedHeight];
    const zs = [z, z + item.packedDepth, z - size.depth, z + item.packedDepth - size.depth];
    for (const cx of xs) for (const cy of ys) for (const cz of zs) points.push([cx, cy, cz]);
  }
  const unique = new Map<string, [number, number, number]>();
  for (const point of points) {
    if (!inside(truck, point, size)) continue;
    unique.set(point.map((value) => value.toFixed(5)).join(":"), point);
  }
  return [...unique.values()];
}

function packOrder(truck: Dimensions, items: PackingItem[]) {
  const packed: PlacedItem[] = [];
  const unfit: PackingItem[] = [];
  for (const item of items) {
    let best: { score: number; position: [number, number, number]; size: Dimensions } | null = null;
    for (const size of rotations(item)) {
      for (const position of candidates(truck, packed, size)) {
        if (packed.some((placed) => intersects(placed, position, size))) continue;
        const score = contactScore(truck, packed, position, size);
        if (!best || score > best.score) best = { score, position, size };
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
      position: best.position,
    });
  }
  const volume = packed.reduce((sum, item) => sum + item.packedWidth * item.packedHeight * item.packedDepth, 0);
  return { packed, unfit, utilization: volume / (truck.width * truck.height * truck.depth) };
}

function packItems(truck: Dimensions, items: PackingItem[]) {
  const orders = [
    [...items].sort((a, b) => b.width * b.height * b.depth - a.width * a.height * a.depth),
    [...items].sort((a, b) => b.height - a.height),
    [...items].sort((a, b) => b.width - a.width),
    [...items].sort((a, b) => b.depth - a.depth),
    [...items].sort((a, b) => b.width * b.depth - a.width * a.depth),
  ];
  return orders
    .map((order) => packOrder(truck, order))
    .sort((a, b) => b.packed.length - a.packed.length || b.utilization - a.utilization)[0] ?? {
      packed: [], unfit: items, utilization: 0,
    };
}

function VehicleModel({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const model = useMemo(() => {
    const clone = scene.clone(true);
    clone.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      child.material = materials.map((source) => {
        const material = source.clone();
        material.transparent = true;
        material.opacity = 0.28;
        material.depthWrite = false;
        return material;
      });
      child.renderOrder = -2;
    });
    return clone;
  }, [scene]);
  return <primitive object={model} />;
}

function PackedItemMesh({ item, truck, active }: { item: PlacedItem; truck: TruckModel; active: boolean }) {
  const { scene } = useGLTF(item.url);
  const prepared = useMemo(() => {
    const model = scene.clone(true);
    const bounds = new THREE.Box3().setFromObject(model);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    model.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      child.material = Array.isArray(child.material)
        ? child.material.map((material) => material.clone())
        : child.material.clone();
    });
    return {
      model,
      center,
      scale: new THREE.Vector3(
        (item.packedWidth * 0.88) / Math.max(size.x, 0.0001),
        (item.packedHeight * 0.88) / Math.max(size.y, 0.0001),
        (item.packedDepth * 0.88) / Math.max(size.z, 0.0001),
      ),
    };
  }, [scene, item.packedWidth, item.packedHeight, item.packedDepth]);
  const [cargoX, cargoFloor, cargoZ] = truck.cargoOffset;
  const [x, y, z] = item.position;
  const position: [number, number, number] = [
    cargoX + x + item.packedWidth / 2 - truck.dimensions.width / 2,
    cargoFloor + y + item.packedHeight / 2,
    cargoZ + z + item.packedDepth / 2 - truck.dimensions.depth / 2,
  ];
  return (
    <group position={position} scale={active ? 1.035 : 1}>
      <group scale={prepared.scale.toArray()}>
        <primitive object={prepared.model} position={prepared.center.clone().multiplyScalar(-1).toArray()} />
      </group>
      <mesh>
        <boxGeometry args={[item.packedWidth, item.packedHeight, item.packedDepth]} />
        <meshBasicMaterial color={item.color} transparent opacity={active ? 0.08 : 0.025} depthWrite={false} />
        <Edges color={active ? "#e56f2e" : item.color} lineWidth={active ? 2.4 : 1.35} />
      </mesh>
    </group>
  );
}

function CameraRig({ truck }: { truck: TruckModel }) {
  const { camera } = useThree();
  const controls = useRef<any>(null);
  useEffect(() => {
    const [, cargoFloor, cargoZ] = truck.cargoOffset;
    camera.position.set(3.9, 2.7, 4.25);
    controls.current?.target.set(0, cargoFloor + truck.dimensions.height * 0.45, cargoZ);
    controls.current?.update();
  }, [camera, truck]);
  return <OrbitControls ref={controls} enablePan={false} minDistance={2} maxDistance={10} />;
}

function PackingScene({ truck, packed, hovered }: { truck: TruckModel; packed: PlacedItem[]; hovered: string | null }) {
  const { width, height, depth } = truck.dimensions;
  const [cargoX, cargoFloor, cargoZ] = truck.cargoOffset;
  return (
    <Canvas camera={{ position: [3.9, 2.7, 4.25], fov: 42 }} dpr={[1, 1.75]}>
      <color attach="background" args={["#f7faf9"]} />
      <ambientLight intensity={1.3} />
      <hemisphereLight args={["#ffffff", "#9daaa5", 1.1]} />
      <directionalLight position={[5, 9, 7]} intensity={1.65} />
      <CameraRig truck={truck} />
      <gridHelper args={[10, 20, "#d2dfda", "#ebf1ef"]} position={[0, -0.008, 0]} />
      <Suspense fallback={null}>
        <VehicleModel url={truck.modelUrl} />
        <group position={[cargoX, cargoFloor + height / 2, cargoZ]}>
          <mesh>
            <boxGeometry args={[width, height, depth]} />
            <meshBasicMaterial color="#79d9b5" transparent opacity={0.035} depthWrite={false} />
            <Edges color="#397c65" lineWidth={1.8} />
          </mesh>
        </group>
        {packed.map((item) => (
          <PackedItemMesh key={item.id} item={item} truck={truck} active={hovered === item.id} />
        ))}
      </Suspense>
    </Canvas>
  );
}

export function TruckFitting({ items }: { items: PackingItem[] }) {
  const [truckId, setTruckId] = useState("cargo-van");
  const [hovered, setHovered] = useState<string | null>(null);
  const truck = TRUCK_MODELS.find((item) => item.id === truckId) ?? TRUCK_MODELS[2];
  const result = useMemo(() => packItems(truck.dimensions, items), [items, truck]);
  const percentage = Math.round(result.utilization * 100);

  return (
    <div className="truck-fit">
      <div className="truck-fit-controls">
        <label>
          <span>Vehicle</span>
          <select value={truckId} onChange={(event) => setTruckId(event.target.value)}>
            {TRUCK_MODELS.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
          </select>
        </label>
        <div className="truck-fit-summary">
          <strong>{result.packed.length}/{items.length} fit</strong>
          <span>{percentage}% volume</span>
        </div>
      </div>
      <div className="truck-fit-canvas">
        <PackingScene truck={truck} packed={result.packed} hovered={hovered} />
      </div>
      <div className="truck-manifest" aria-label="Packed items">
        {items.map((item) => {
          const fits = result.packed.some((packed) => packed.id === item.id);
          return (
            <div key={item.id} className={fits ? "" : "unfit"} onMouseEnter={() => setHovered(item.id)} onMouseLeave={() => setHovered(null)}>
              <i style={{ background: item.color }} />
              <span>{item.name}</span>
              <small>{item.width.toFixed(2)} × {item.height.toFixed(2)} × {item.depth.toFixed(2)} m</small>
              <b>{fits ? "Packed" : "Doesn’t fit"}</b>
            </div>
          );
        })}
      </div>
    </div>
  );
}
