import * as THREE from 'three';
// import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import metaversefile from 'metaversefile';
const {useFrame, useMaterials, useRenderer, useCamera, useProcGen, useLocalPlayer, useMathUtils} = metaversefile;

const localVector = new THREE.Vector3();
const localQuaternion = new THREE.Quaternion();

const upVector = new THREE.Vector3(0, 1, 0);

const radiusTop = 0.01;
const radiusBottom = radiusTop;
const height = 0.8;
const playerCenterRadius = 0.3;
const radialSegments = 8;
const heightSegments = 8;
const openEnded = false;
const segmentLength = 0.05;
const verticesPerHeightSegment = radialSegments + 1;
const verticesPerPart = verticesPerHeightSegment * (heightSegments + 1);
const numPointSets = 2;
const numBlades = 8 * 1024;
const range = 5;

const createSilkGrassBladeGeometry = () => {
  const geometryNonInstanced = new THREE.CylinderGeometry(
    radiusTop,
    radiusBottom,
    height,
    radialSegments,
    heightSegments,
    openEnded,
  ).translate(0, height/2, 0);
  const geometry = new THREE.InstancedBufferGeometry();
  for (const k in geometryNonInstanced.attributes) {
    geometry.setAttribute(k, geometryNonInstanced.attributes[k]);
  }
  geometry.index = geometryNonInstanced.index;
  return geometry;
};
function createSilksGeometry() {
  const geometry = createSilkGrassBladeGeometry();
  /* geometry.setAttribute('p', new THREE.InstancedBufferAttribute(new Float32Array(maxParticles * 3), 3));
  // geometry.setAttribute('q', new THREE.InstancedBufferAttribute(new Float32Array(maxParticles * 4), 4));
  geometry.setAttribute('t', new THREE.InstancedBufferAttribute(new Float32Array(maxParticles * 2), 2));
  geometry.setAttribute('textureIndex', new THREE.InstancedBufferAttribute(new Int32Array(maxParticles), 1)); */
  geometry.setAttribute(
    'p',
    new THREE.InstancedBufferAttribute(new Float32Array(numBlades * 3), 3)
  );
  geometry.setAttribute(
    'q',
    new THREE.InstancedBufferAttribute(new Float32Array(numBlades * 4), 4)
  );

  const procGen = useProcGen();
  const {alea} = procGen;
  const rng = alea('lol');
  const r = n => -n + rng() * 2 * n;
  for (let i = 0; i < numBlades; i++) {
    localVector.set(r(5), 0, r(5))
      .toArray(geometry.attributes.p.array, i * 3);
    localQuaternion.setFromAxisAngle(upVector, r(Math.PI))
      .toArray(geometry.attributes.q.array, i * 4);
  }
  return geometry;
};
const _makeSilksMesh = () => {
  const {WebaverseShaderMaterial} = useMaterials();
  
  const geometry = createSilksGeometry();

  const _makeRenderTarget = () => new THREE.WebGLRenderTarget(512, 512, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
    type: THREE.FloatType,
    wrapS: THREE.RepeatWrapping,
    wrapT: THREE.RepeatWrapping,
    stencilBuffer: false,
  });
  const displacementMaps = [
    _makeRenderTarget(),
    _makeRenderTarget(),
  ];
  const displacementMapScene = (() => {
    const fullScreenQuadGeometry = new THREE.PlaneBufferGeometry(2, 2);
    const fullscreenVertexShader = `\
      varying vec2 vUv;

      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 1.0, 1.0);
      }
    `;
    const fullscreenFragmentShader = `\
      uniform vec3 uPlayerPosition;
      uniform vec3 uWorldPosition;
      uniform sampler2D uDisplacementMap;
      varying vec2 vUv;

      const float range = ${range.toFixed(8)};
      const float learningRate = 0.005;
      const float maxDistance = 0.6;

      // optimized distance to line segment function (capsule shape with rounded caps)
      float distanceToLine(vec3 p, vec3 pointA, vec3 pointB) {
        vec3 v = pointB - pointA;
        vec3 w = p - pointA;
        float c1 = dot(w, v);
        if (c1 <= 0.0) {
          return length(p - pointA);
        }
        float c2 = dot(v, v);
        if (c2 <= c1) {
          return length(p - pointB);
        }
        float b = c1 / c2;
        vec3 pointOnLine = pointA + b * v;
        return length(p - pointOnLine);
      }

      void main() {
        vec2 virtualXZ = vec2(vUv.x * 2.0 - 1.0, vUv.y * 2.0 - 1.0) * range;
        virtualXZ += uWorldPosition.xz;
        float distanceToPlayer = distanceToLine(
          vec3(virtualXZ.x, 0., virtualXZ.y),
          uPlayerPosition,
          uPlayerPosition + vec3(0., -1.5, 0.)
        );

        // float distanceToPlayer = length(virtualXZ - uPlayerPosition.xz);
        vec2 direction = distanceToPlayer > 0.0 ? normalize(virtualXZ - uPlayerPosition.xz) : vec2(0.0, 0.0);

        vec3 oldColor = texture2D(uDisplacementMap, vUv).rgb;

        vec3 newColor = vec3(direction, 0.);
        float distanceFactor = min(max(maxDistance - distanceToPlayer, 0.), 1.);
        
        float localLearningRate = learningRate;
        if (distanceFactor > 0.0) {
          localLearningRate = 1.;
        }
        // gl_FragColor = vec4(newColor * localLearningRate, 1.);
        gl_FragColor = vec4(
          min(max(
            oldColor * (1. - learningRate) +
              (newColor * distanceFactor) * localLearningRate,
          vec3(-1.)), vec3(1.0)),
          1.
        );
      }
    `;
    const fullscreenMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uPlayerPosition: {
          value: new THREE.Vector3(0, 0, 0),
          needsUpdate: false,
        },
        uWorldPosition: {
          value: new THREE.Vector3(0, 0, 0),
          needsUpdate: false,
        },
        uDisplacementMap: {
          value: displacementMaps[0].texture,
          needsUpdate: false,
        },
      },
      vertexShader: fullscreenVertexShader,
      fragmentShader: fullscreenFragmentShader,
      // side: THREE.DoubleSide,
    });
    const fullscreenQuadMesh = new THREE.Mesh(fullScreenQuadGeometry, fullscreenMaterial);
    fullscreenQuadMesh.frustumCulled = false;
    const scene = new THREE.Scene();
    scene.add(fullscreenQuadMesh);
    scene.update = () => {
      const localPlayer = useLocalPlayer();
      fullscreenMaterial.uniforms.uPlayerPosition.value.copy(localPlayer.position);
      fullscreenMaterial.uniforms.uPlayerPosition.needsUpdate = true;

      fullscreenMaterial.uniforms.uWorldPosition.value.setFromMatrixPosition(mesh.matrixWorld);
      fullscreenMaterial.uniforms.uWorldPosition.needsUpdate = true;

      fullscreenMaterial.uniforms.uDisplacementMap.value = displacementMaps[0].texture;
      fullscreenMaterial.uniforms.uDisplacementMap.needsUpdate = true;
    };
    return scene;
  })();
  const _renderDisplacementMap = () => {
    const renderer = useRenderer();
    const camera = useCamera();

    {
      // push state
      const oldRenderTarget = renderer.getRenderTarget();
    
      // update
      displacementMapScene.update();

      // render
      renderer.setRenderTarget(displacementMaps[1]);
      renderer.clear();
      renderer.render(displacementMapScene, camera);

      // pop state
      renderer.setRenderTarget(oldRenderTarget);
    }
  };

  const material = new WebaverseShaderMaterial({
    uniforms: {
      uTime: {
        type: 'f',
        value: 0,
        needsUpdate: true,
      },
      uDisplacementMap: {
        type: 't',
        value: displacementMaps[1].texture,
        needsUpdate: true,
      },
    },
    vertexShader: `\
      precision highp float;
      precision highp int;

      uniform float uTime;
      uniform sampler2D uDisplacementMap;
      attribute vec3 p;
      attribute vec4 q;
      varying vec2 vUv;
      varying vec2 vUv2;

      vec4 quat_from_axis_angle(vec3 axis, float angle) { 
        vec4 qr;
        float half_angle = (angle * 0.5);
        qr.x = axis.x * sin(half_angle);
        qr.y = axis.y * sin(half_angle);
        qr.z = axis.z * sin(half_angle);
        qr.w = cos(half_angle);
        return qr;
      }

      vec3 rotate_vertex_position(vec3 position, vec4 q) { 
        vec3 v = position.xyz;
        return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
      }

      void main() {
        vec3 pos = position;
        vUv = uv;
        vUv2 = (p.xz + ${(range).toFixed(8)}) / ${(range * 2).toFixed(8)};
        
        // instance
        {
          pos = rotate_vertex_position(pos, q);
          pos += p;
        }

        // displacement
        {
          vec3 displacement = texture2D(uDisplacementMap, vUv2).rgb;
          pos.xz += displacement.xy * pow(pos.y, 0.5) * 0.5;
        }
      
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `\
      precision highp float;
      precision highp int;

      #define PI 3.1415926535897932384626433832795

      uniform float uTime;
      uniform sampler2D uDisplacementMap;
      varying float vOffset;
      varying vec2 vUv;
      varying vec2 vUv2;

      vec3 hueShift( vec3 color, float hueAdjust ){
        const vec3  kRGBToYPrime = vec3 (0.299, 0.587, 0.114);
        const vec3  kRGBToI      = vec3 (0.596, -0.275, -0.321);
        const vec3  kRGBToQ      = vec3 (0.212, -0.523, 0.311);

        const vec3  kYIQToR     = vec3 (1.0, 0.956, 0.621);
        const vec3  kYIQToG     = vec3 (1.0, -0.272, -0.647);
        const vec3  kYIQToB     = vec3 (1.0, -1.107, 1.704);

        float   YPrime  = dot (color, kRGBToYPrime);
        float   I       = dot (color, kRGBToI);
        float   Q       = dot (color, kRGBToQ);
        float   hue     = atan (Q, I);
        float   chroma  = sqrt (I * I + Q * Q);

        hue += hueAdjust;

        Q = chroma * sin (hue);
        I = chroma * cos (hue);

        vec3    yIQ   = vec3 (YPrime, I, Q);

        return vec3( dot (yIQ, kYIQToR), dot (yIQ, kYIQToG), dot (yIQ, kYIQToB) );
      }

      void main() {
        float t = pow(uTime, 0.5)/2. + 0.5;

        gl_FragColor = vec4(0., 0., 0., 1.);
        gl_FragColor += texture2D(uDisplacementMap, vUv2);
      }
    `,
  });
  const mesh = new THREE.InstancedMesh(geometry, material, numBlades);
  mesh.frustumCulled = false;

  mesh.update = (timestamp, timeDiff) => {
    const maxTime = 3000;
    const f = (timestamp % maxTime) / maxTime;

    _renderDisplacementMap();

    material.uniforms.uTime.value = f;
    material.uniforms.uTime.needsUpdate = true;

    material.uniforms.uDisplacementMap.value = displacementMaps[1].texture;
    material.uniforms.uDisplacementMap.needsUpdate = true;

    {
      const temp = displacementMaps[0];
      displacementMaps[0] = displacementMaps[1];
      displacementMaps[1] = temp;
    }
  };
  return mesh;
};
export default () => {
  const mesh = _makeSilksMesh();

  useFrame(({timestamp, timeDiff}) => {
    mesh.update(timestamp, timeDiff);
  });
  
  return mesh;
};