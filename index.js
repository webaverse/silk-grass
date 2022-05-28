import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import metaversefile from 'metaversefile';
const {useFrame, useApp, useScene, useSound, useMaterials, useRenderer, useCamera, useProcGen, useLocalPlayer, useHitManager, useLodder} = metaversefile;

const baseUrl = import.meta.url.replace(/(\/)[^\/\\]*$/, '$1');

const localVector = new THREE.Vector3();
const localQuaternion = new THREE.Quaternion();
const localEuler = new THREE.Euler();
const localVector2D = new THREE.Vector2();
// const localVector2D2 = new THREE.Vector2();
// const localBox2D = new THREE.Box2();

const zeroVector = new THREE.Vector3(0, 0, 0);
const upVector = new THREE.Vector3(0, 1, 0);
const gravity = new THREE.Vector3(0, -9.8, 0);
const dropItemSize = 0.2;
const chunkWorldSize = 10;
const numLods = 1;

const height = 0.8;
const radialSegments = 3;
const heightSegments = 8;
const openEnded = false;
const segmentHeight = height / heightSegments;
const numBlades = 8 * 1024;
const cutTime = 1;
const growTime = 60;
const cutGrowTime = cutTime + growTime;
const cutHeightOffset = -1.4;
const floorLimit = dropItemSize / 2;

const windRotation = ((Date.now() / 1000) % 1) * Math.PI * 2;

//

const itemletImageUrls = [
  /* 'noun-fruit-4617781.svg',
  'noun-root-4617773.svg',
  'noun-potion-3097512.svg',
  'noun-poison-4837113.svg', */
  'HP-01.svg',
  'HP_negative-01.svg',
  'MP_green-01.svg',
  'MP_purple-01.svg',
  'XP-01.svg',
  'XP_negative-01.svg',
].map(name => `/images/items/${name}`);
/* const colorPalettes = [
  [ // red
    0xef5350,
    0xd32f2f,
  ],
  [ // violet
    0x7e57c2,
    0x512da8,
  ],
  [ // green
    0x66bb6a,
    0x388e3c,
  ],
  [ // grey
    0xbdbdbd,
    0x455a64,
  ],
].map(([color1, color2]) => {
  color1 = new THREE.Color(color1).offsetHSL(0, 0.5, 0);
  color2 = new THREE.Color(color2).offsetHSL(0, 0.5, 0);

  return [
    color1.getHex(),
    color2.getHex(),
  ]
}); */
const _averagePoints = (points, target) => {
  target.copy(points[0]);
  for (let i = 1; i < points.length; i++) {
    target.add(points[i]);
  }
  return target.divideScalar(points.length);
};

//

const fullScreenQuadGeometry = new THREE.PlaneBufferGeometry(2, 2);
const fullscreenVertexShader = `\
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 1.0, 1.0);
  }
`;

const makeNoiseTexture = (size = 256) => {
  const procGen = useProcGen();
  const {alea} = procGen;
  const rng = alea('noise');
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  const imageData = context.getImageData(0, 0, size, size);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = rng() * 255;
    data[i + 1] = rng() * 255;
    data[i + 2] = rng() * 255;
    data[i + 3] = rng() * 255;
  }
  context.putImageData(imageData, 0, 0);
  const texture = new THREE.Texture(
    canvas,
    THREE.UVMapping,
    THREE.RepeatWrapping,
    THREE.RepeatWrapping,
  );
  texture.needsUpdate = true;
  return texture;
};
const getNoiseTexture = (() => {
  let noiseTexture = null;
  return () => {
    if (!noiseTexture) {
      noiseTexture = makeNoiseTexture();
    }
    return noiseTexture;
  };
})();
const makeSeamlessNoiseTexture = () => {
  const img = new Image();
  const texture = new THREE.Texture(img);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;

  img.crossOrigin = 'Anonymous';
  img.onload = () => {
    // console.log('load image', img);
    // document.body.appendChild(img);
    texture.needsUpdate = true;
  };
  img.onerror = err => {
    console.warn(err);
  };
  img.src = `${baseUrl}perlin-noise.jpg`;

  return texture;
};
const getSeamlessNoiseTexture = (() => {
  let noiseTexture = null;
  return () => {
    if (!noiseTexture) {
      noiseTexture = makeSeamlessNoiseTexture();
    }
    return noiseTexture;
  };
})();

const createSilkGrassBladeGeometry = () => {
  const geometryNonInstanced = (() => {
    const radiusTop = 0.05;
    const radiusBottom = radiusTop;
    const baseGeometry = new THREE.CylinderGeometry(
      radiusTop,
      radiusBottom,
      segmentHeight,
      radialSegments,
      1, // heightSegments,
      openEnded,
    );
    baseGeometry.setAttribute('segment', new THREE.BufferAttribute(new Int32Array(baseGeometry.attributes.position.count), 1));
    const geometries = [];
    for (let i = 0; i < heightSegments; i++) {
      const geometry = baseGeometry.clone()
        .translate(0, segmentHeight/2 + segmentHeight * i, 0);
      geometry.attributes.segment.array.fill(i);
      geometries.push(geometry);
    }
    const result = BufferGeometryUtils.mergeBufferGeometries(geometries);
    for (let i = 0; i < result.attributes.position.count; i++) {
      localVector.fromArray(result.attributes.position.array, i * 3);
      const heightFactor = (height - localVector.y) / height;
      localVector.x *= heightFactor;
      localVector.z *= heightFactor;
      localVector.toArray(result.attributes.position.array, i * 3);
    }
    return result;
  })();
  const geometry = new THREE.InstancedBufferGeometry();
  for (const k in geometryNonInstanced.attributes) {
    geometry.setAttribute(k, geometryNonInstanced.attributes[k]);
  }
  geometry.index = geometryNonInstanced.index;
  return geometry;
};

function createSilksBaseGeometry() {
  const geometry = createSilkGrassBladeGeometry();
  geometry.setAttribute(
    'p',
    new THREE.InstancedBufferAttribute(new Float32Array(numBlades * 3), 3)
  );
  geometry.setAttribute(
    'q',
    new THREE.InstancedBufferAttribute(new Float32Array(numBlades * 4), 4)
  );
  return geometry;
};
const silksBaseGeometry = createSilksBaseGeometry();
const _makeCutMesh = () => {
  // const {WebaverseShaderMaterial} = useMaterials();

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(4 * 3), 3));
  geometry.setIndex(new THREE.BufferAttribute(Uint16Array.from([0, 1, 2, 2, 1, 3]), 1));
  const material = new THREE.MeshBasicMaterial({
    color: 0x0000FF,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  return mesh;
};
const _renderSilksGeometry = geometry => {
  const procGen = useProcGen();
  const {alea} = procGen;
  const rng = alea('lol');
  const r = n => -n + rng() * 2 * n;
  for (let i = 0; i < numBlades; i++) {
    localVector.set(rng() * chunkWorldSize, 0, rng() * chunkWorldSize)
      .toArray(geometry.attributes.p.array, i * 3);
    localQuaternion.setFromAxisAngle(upVector, r(Math.PI))
      .toArray(geometry.attributes.q.array, i * 4);
  }
};
const _makeSilksMesh = () => {
  const {WebaverseShaderMaterial} = useMaterials();

  const geometry = silksBaseGeometry.clone();
  _renderSilksGeometry(geometry);

  const _makeRenderTarget = () => new THREE.WebGLRenderTarget(512, 512, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    format: THREE.RGBAFormat,
    type: THREE.FloatType,
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    stencilBuffer: false,
  });
  const displacementMaps = [
    _makeRenderTarget(),
    _makeRenderTarget(),
  ];
  const displacementMapScene = (() => {
    const fullscreenFragmentShader = `\
      uniform vec3 uPlayerPosition;
      uniform vec3 uWorldPosition;
      uniform sampler2D uNoiseTexture;
      uniform sampler2D uDisplacementMap;
      uniform vec3 pA1;
      uniform vec3 pA2;
      uniform vec3 pB1;
      uniform vec3 pB2;
      varying vec2 vUv;

      const float chunkWorldSize = ${chunkWorldSize.toFixed(8)};
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
        vec2 virtualXZ = vec2(vUv.x, vUv.y) * chunkWorldSize;
        virtualXZ += uWorldPosition.xz;
        float distanceToPlayer = distanceToLine(
          vec3(virtualXZ.x, 0., virtualXZ.y),
          uPlayerPosition,
          uPlayerPosition + vec3(0., -1.5, 0.)
        );

        // float distanceToPlayer = length(virtualXZ - uPlayerPosition.xz);
        vec2 direction = distanceToPlayer > 0.0 ? normalize(virtualXZ - uPlayerPosition.xz) : vec2(0.0, 0.0);

        vec4 oldColor = texture2D(uDisplacementMap, vUv);

        vec4 newColor = vec4(direction, oldColor.zw);
        float distanceFactor = min(max(maxDistance - distanceToPlayer, 0.), 1.);
        
        float localLearningRate = learningRate;
        if (distanceFactor > 0.0) {
          localLearningRate = 1.;
        }
        gl_FragColor = vec4(
          min(max(
            oldColor.xy * (1. - learningRate) +
              (newColor.xy * distanceFactor) * localLearningRate,
          vec2(-1.)), vec2(1.)),
          newColor.z,
          newColor.w
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
        uNoiseTexture: {
          value: getNoiseTexture(),
          needsUpdate: true,
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
  const displacementMapScene2 = (() => {
    const fullscreenFragmentShader2 = `\
      uniform vec3 uWorldPosition;
      uniform sampler2D uDisplacementMap;
      uniform float uTime;
      uniform vec3 pA1;
      uniform vec3 pA2;
      uniform vec3 pB1;
      uniform vec3 pB2;
      varying vec2 vUv;

      const float chunkWorldSize = ${chunkWorldSize.toFixed(8)};
      const float learningRate = 0.005;
      const float maxDistance = 0.6;

      bool isPointInTriangle(vec2 point, vec2 a, vec2 b, vec2 c) {
        vec2 v0 = c - a;
        vec2 v1 = b - a;
        vec2 v2 = point - a;
    
        float dot00 = dot(v0, v0);
        float dot01 = dot(v0, v1);
        float dot02 = dot(v0, v2);
        float dot11 = dot(v1, v1);
        float dot12 = dot(v1, v2);
    
        float invDenom = 1. / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    
        return (u >= 0.) && (v >= 0.) && (u + v < 1.);
      }
      
      const float height = ${height.toFixed(8)};
      const float cutTime = ${cutTime.toFixed(8)};
      const float growTime = ${growTime.toFixed(8)};
      void main() {
        vec2 virtualXZ = vec2(vUv.x, vUv.y) * chunkWorldSize;
        virtualXZ += uWorldPosition.xz;

        vec4 color = texture2D(uDisplacementMap, vUv);

        vec2 a = pA1.xz;
        vec2 b = pA2.xz;
        vec2 c = pB1.xz;
        vec2 d = pB2.xz;
        float timeDiff = uTime - color.w;
        if (
          (
            isPointInTriangle(virtualXZ, a, b, c) || isPointInTriangle(virtualXZ, b, d, c)
          ) &&
          timeDiff > (cutTime + growTime * 0.5)
        ) {
          // color.z = (pA1.y + pA2.y + pB1.y + pB2.y) / 4.;
          color.z = height / 8.;
          color.w = uTime;
        }
        gl_FragColor = color;
      }
    `;
    const fullscreenMaterial2 = new THREE.ShaderMaterial({
      uniforms: {
        uWorldPosition: {
          value: new THREE.Vector3(0, 0, 0),
          needsUpdate: false,
        },
        uDisplacementMap: {
          value: displacementMaps[0].texture,
          needsUpdate: false,
        },
        uNoiseTexture: {
          value: getNoiseTexture(),
          needsUpdate: true,
        },
        uTime: {
          value: 0,
          needsUpdate: false,
        },
        pA1: {
          value: new THREE.Vector3(),
          needsUpdate: false,
        },
        pA2: {
          value: new THREE.Vector3(),
          needsUpdate: false,
        },
        pB1: {
          value: new THREE.Vector3(),
          needsUpdate: false,
        },
        pB2: {
          value: new THREE.Vector3(),
          needsUpdate: false,
        },
      },
      vertexShader: fullscreenVertexShader,
      fragmentShader: fullscreenFragmentShader2,
      // side: THREE.DoubleSide,
    });
    const fullscreenQuadMesh2 = new THREE.Mesh(fullScreenQuadGeometry, fullscreenMaterial2);
    fullscreenQuadMesh2.frustumCulled = false;
    const scene2 = new THREE.Scene();
    scene2.add(fullscreenQuadMesh2);
    scene2.update = (pA1, pA2, pB1, pB2, timestamp) => {
      fullscreenMaterial2.uniforms.uWorldPosition.value.setFromMatrixPosition(mesh.matrixWorld);
      fullscreenMaterial2.uniforms.uWorldPosition.needsUpdate = true;

      fullscreenMaterial2.uniforms.uDisplacementMap.value = displacementMaps[0].texture;
      fullscreenMaterial2.uniforms.uDisplacementMap.needsUpdate = true;

      fullscreenMaterial2.uniforms.pA1.value.copy(pA1);
      fullscreenMaterial2.uniforms.pA1.needsUpdate = true;
      fullscreenMaterial2.uniforms.pA2.value.copy(pA2);
      fullscreenMaterial2.uniforms.pA2.needsUpdate = true;
      fullscreenMaterial2.uniforms.pB1.value.copy(pB1);
      fullscreenMaterial2.uniforms.pB1.needsUpdate = true;
      fullscreenMaterial2.uniforms.pB2.value.copy(pB2);
      fullscreenMaterial2.uniforms.pB2.needsUpdate = true;

      const timestampS = timestamp / 1000;
      fullscreenMaterial2.uniforms.uTime.value = timestampS;
      fullscreenMaterial2.uniforms.uTime.needsUpdate = true;
    };
    return scene2;
  })();
  const _renderDisplacementMap = () => {
    const renderer = useRenderer();
    const context = renderer.getContext();
    const camera = useCamera();

    {
      // update
      displacementMapScene.update();

      // push state
      const oldRenderTarget = renderer.getRenderTarget();
      context.disable(context.SAMPLE_ALPHA_TO_COVERAGE);

      // render
      renderer.setRenderTarget(displacementMaps[1]);
      renderer.clear();
      renderer.render(displacementMapScene, camera);

      // pop state
      renderer.setRenderTarget(oldRenderTarget);
      context.enable(context.SAMPLE_ALPHA_TO_COVERAGE);
    }
  };
  const _renderCut = (pA1, pA2, pB1, pB2, timestamp) => {
    const renderer = useRenderer();
    const context = renderer.getContext();
    const camera = useCamera();

    {
      // update
      displacementMapScene2.update(pA1, pA2, pB1, pB2, timestamp);
      
      // push state
      const oldRenderTarget = renderer.getRenderTarget();
      context.disable(context.SAMPLE_ALPHA_TO_COVERAGE);

      // render
      renderer.setRenderTarget(displacementMaps[1]);
      renderer.clear();
      renderer.render(displacementMapScene2, camera);

      // pop state
      renderer.setRenderTarget(oldRenderTarget);
      context.enable(context.SAMPLE_ALPHA_TO_COVERAGE);
    }
  };
  const _renderMain = timestamp => {
    const timestampS = timestamp / 1000;
    material.uniforms.uTime.value = timestampS;
    material.uniforms.uTime.needsUpdate = true;

    material.uniforms.uDisplacementMap.value = displacementMaps[1].texture;
    material.uniforms.uDisplacementMap.needsUpdate = true;
  };
  const _flipRenderTargets = () => {
    const temp = displacementMaps[0];
    displacementMaps[0] = displacementMaps[1];
    displacementMaps[1] = temp;
  };

  const material = new WebaverseShaderMaterial({
    uniforms: {
      uTime: {
        value: 0,
        needsUpdate: true,
      },
      uDisplacementMap: {
        value: displacementMaps[1].texture,
        needsUpdate: true,
      },
      uNoiseTexture: {
        value: getNoiseTexture(),
        needsUpdate: true,
      },
      uSeamlessNoiseTexture: {
        value: getSeamlessNoiseTexture(),
        needsUpdate: true,
      },
      uWindRotation: {
        value: windRotation,
        needsUpdate: true,
      },
    },
    vertexShader: `\
      precision highp float;
      precision highp int;

      uniform float uTime;
      uniform sampler2D uDisplacementMap;
      uniform sampler2D uNoiseTexture;
      uniform sampler2D uSeamlessNoiseTexture;
      uniform float uWindRotation;
      attribute vec3 p;
      attribute vec4 q;
      attribute int segment;
      varying vec2 vUv;
      varying vec2 vUv2;
      varying vec3 vNormal;
      varying float vTimeDiff;
      varying float vY;
      varying vec3 vNoise;

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
        return position + 2.0 * cross(q.xyz, cross(q.xyz, position) + q.w * position);
      }

      const float segmentHeight = ${segmentHeight.toFixed(8)};
      const float heightSegments = ${heightSegments.toFixed(8)};  
      const float topSegmentY = segmentHeight * heightSegments;
      const float cutTime = ${cutTime.toFixed(8)};
      const float growTime = ${growTime.toFixed(8)};
      const float cutGrowTime = ${cutGrowTime.toFixed(8)};
      void main() {
        vec3 pos = position;
        vUv = uv;
        vUv2 = p.xz / ${chunkWorldSize.toFixed(8)};

        // time diff
        vec4 displacementColor = texture2D(uDisplacementMap, vUv2);
        vTimeDiff = uTime - displacementColor.w;

        // cut handling
        float segmentStartY = float(segment) * segmentHeight;
        float cutY = displacementColor.z;
        float cutSegmentY = floor(cutY / segmentHeight) * segmentHeight;
        bool isCuttableY = (cutY > 0. && cutY < segmentStartY);
        bool isCut = isCuttableY && (vTimeDiff < cutTime);
        bool isGrow = isCuttableY && (vTimeDiff >= cutTime && vTimeDiff < cutGrowTime);
        if (isCut) {
          vec3 centerOfBlade = vec3(0., (cutSegmentY + topSegmentY) * 0.5, 0.);
          // float scaleFactor = max(1. - vTimeDiff / cutTime, 0.);

          // acceleration / velocity
          const float GRAVITY = -9.8;
          const float CUT_VELOCITY = 2.;

          vec2 vUv3 = mod(vUv2 * 3., 1.); // to not conflict with rotation axis
          vec2 directionXZ = -1. + texture2D(uNoiseTexture, vUv3).xz * 2.;
          
          // compute the acceleration / velocity offset
          vec3 gravityVector = vec3(0., GRAVITY * vTimeDiff * vTimeDiff * 0.5, 0.);
          float cutSpeed = texture2D(uNoiseTexture, vUv2).w * 3.;
          vec3 direction = vec3(directionXZ.x, CUT_VELOCITY, directionXZ.y);
          vec3 velocityVector = direction * vTimeDiff * cutSpeed;

          float centerOfBladePositionY = centerOfBlade.y + velocityVector.y + gravityVector.y;
          if (centerOfBladePositionY >= 0.) {
            // scale + rotation
            pos -= centerOfBlade;
            vec3 rotationAxis = normalize(-1. + texture2D(uNoiseTexture, vUv2).xyz * 2.);
            vec4 q = quat_from_axis_angle(rotationAxis, uTime * 2. * PI * 0.2);
            pos = rotate_vertex_position(pos, q);
            pos += centerOfBlade;

            // velocity + position
            pos += velocityVector + gravityVector;
          } else {
            pos = vec3(0.);
          }
        } else if (isGrow) {
          vec3 bottomOfBlade = vec3(0., cutSegmentY * 0.5, 0.);
          float scaleFactor = min((vTimeDiff - cutTime) / growTime, 1.);

          // grow
          pos -= bottomOfBlade;
          pos.y *= scaleFactor;
          pos += bottomOfBlade;
        }

        // instance offset
        {
          pos = rotate_vertex_position(pos, q);
          pos += p;
        }

        if (!isCut) {
          // wind
          float windOffsetX = texture2D(
            uSeamlessNoiseTexture,
            (vUv2 * 0.1) * 3. +
              vec2(uTime * 0.05)
          ).r * pos.y;
          /* float windOffsetY = texture2D(
            uSeamlessNoiseTexture,
            (vUv2 * 0.03) * 3. + 1. +
              vec2(uTime * 0.001)
          ).r;
          float windOffsetZ = texture2D(
            uSeamlessNoiseTexture,
            (vUv2 * 0.03) * 3. + 2. +
              vec2(uTime * 0.001)
          ).r; */
          float windOffsetY = 0.;
          float windOffsetZ = 0.;
          vec3 windOffset = vec3(windOffsetX, windOffsetY, windOffsetZ);
          vec4 q2 = quat_from_axis_angle(vec3(0., 1., 0.), uWindRotation);
          windOffset = rotate_vertex_position(windOffset, q2);
          pos += windOffset * 1.;
        }

        vY = pos.y;
        vNoise = texture2D(uNoiseTexture, vUv2).xyz;

        // displacement bend
        if (!isCut) {
          vec4 displacement = texture2D(uDisplacementMap, vUv2);
          pos.xz += displacement.xy * pow(pos.y, 0.5) * 0.5;
        }

        // output
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        
        vNormal = normal;
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
      varying vec3 vNormal;
      varying float vTimeDiff;
      varying float vY;
      varying vec3 vNoise;

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

      float rand(float n){return fract(sin(n) * 43758.5453123);}

      const float height = ${height.toFixed(8)};
      const float cutTime = ${cutTime.toFixed(8)};
      const vec3 color = vec3(${new THREE.Color(0x66bb6a).toArray().join(', ')});
      void main() {
        vec4 displacementColor = texture2D(uDisplacementMap, vUv2);
        
        // gl_FragColor.rgb = displacementColor.rgb;
        gl_FragColor.rgb = color *
          (0.4 + rand(floor(100. + (vNoise.x + vNoise.y + vNoise.z) * 15.)) * 0.6) *
          (0.2 + vY/height * 0.8);
        gl_FragColor.a = 1.;
      }
    `,
    // transparent: true,
  });
  const mesh = new THREE.InstancedMesh(geometry, material, numBlades);
  mesh.frustumCulled = false;

  mesh.update = (timestamp, timeDiff) => {
    _renderDisplacementMap();
    _renderMain(timestamp);
    _flipRenderTargets();
  };

  /* // XXX debugging
  {
    const scene = useScene();
    const cutMesh = _makeCutMesh();
    cutMesh.frustumCulled = false;
    scene.add(cutMesh);
  } */

  const cutLastTimestampMap = new Float32Array(chunkWorldSize ** 2);
  mesh.hitAttempt = (position, quaternion, target2D) => {
    const pointA1 = position.clone()
      .add(new THREE.Vector3(-1, cutHeightOffset, -0.1).applyQuaternion(quaternion));
    const pointA2 = position.clone()
      .add(new THREE.Vector3(-0.7, cutHeightOffset, -1.5).applyQuaternion(quaternion));
    const pointB1 = position.clone()
      .add(new THREE.Vector3(1, cutHeightOffset, -0.1).applyQuaternion(quaternion));
    const pointB2 = position.clone()
      .add(new THREE.Vector3(0.7, -1.2, -1.5).applyQuaternion(quaternion));
    /* [pointA1, pointA2, pointB1, pointB2].forEach((point, i) => {
      point.toArray(cutMesh.geometry.attributes.position.array, i * 3);
    });
    cutMesh.geometry.attributes.position.needsUpdate = true; */

    const timestamp = performance.now();
    _renderCut(pointA1, pointA2, pointB1, pointB2, timestamp);
    _flipRenderTargets();

    const points = [
      pointA1,
      pointA2,
      pointB1,
      pointB2,
    ];
    const hitCenterPoint = _averagePoints(points, new THREE.Vector3());
    const relativeX = Math.floor(hitCenterPoint.x);
    const relativeZ = Math.floor(hitCenterPoint.z);    

    const meshWorldPosition = new THREE.Vector3().setFromMatrixPosition(mesh.matrixWorld);
    const meshWorldMin = meshWorldPosition.clone().add(new THREE.Vector3(0, 0, 0));
    const meshWorldMax = meshWorldPosition.clone().add(new THREE.Vector3(chunkWorldSize, 0, chunkWorldSize));
    if (
      relativeX >= meshWorldMin.x && relativeZ >= meshWorldMin.z &&
      relativeX < meshWorldMax.x && relativeZ < meshWorldMax.z
    ) {
      const localX = relativeX - meshWorldMin.x;
      const localZ = relativeZ - meshWorldMin.z;
      const index = localX + localZ * chunkWorldSize;
      const timeDiff = timestamp - cutLastTimestampMap[index];
      if (timeDiff >= (cutTime + growTime / 2) * 1000) {
        cutLastTimestampMap[index] = timestamp;
        return target2D.set(relativeX + Math.random(), relativeZ + Math.random());
      } else {
        return null;
      }
    } else {
      return null;
    }
  };
  return mesh;
};

class GrassChunkGenerator {
  constructor(parent) {
    // parameters
    this.parent = parent;

    // mesh
    this.chunks = new THREE.Group();
  }
  getMeshes() {
    return this.chunks.children;
  }
  generateChunk(chunk) {
    const mesh = _makeSilksMesh();
    mesh.position.set(chunk.x * chunkWorldSize, 0, chunk.z * chunkWorldSize);
    this.chunks.add(mesh);
    mesh.updateMatrixWorld();

    chunk.binding = mesh;
  }
  disposeChunk(chunk) {
    const mesh = chunk.binding;
    this.chunks.remove(mesh);
    chunk.binding = null;
  }
  update(timestamp, timeDiff) {
    for (const mesh of this.getMeshes()) {
      mesh.update(timestamp, timeDiff);
    }
  }
  destroy() {
    // nothing; the owning lod tracker disposes of our contents
  }
}

export default e => {
  const app = useApp();
  const scene = useScene();
  const {WebaverseShaderMaterial} = useMaterials();
  const hitManager = useHitManager();
  const {LodChunkTracker} = useLodder();
  const sounds = useSound();

  app.name = 'silk-grass';

  const generator = new GrassChunkGenerator(this);
  const tracker = new LodChunkTracker(generator, {
    chunkWorldSize,
    numLods,
  });

  app.add(generator.chunks);
  generator.chunks.updateMatrixWorld();

  useFrame(() => {
    const localPlayer = useLocalPlayer();
    tracker.update(localPlayer.position);
  });

  // itemlets support
  // XXX this should be a type of drop in the drop manager
  let itemletTextures = null;
  e.waitUntil((async () => {
    itemletTextures = await Promise.all(itemletImageUrls.map(url => {
      return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = async () => {
          const imageBitmap = await createImageBitmap(image, {
            imageOrientation: 'flipY',
          });
          const texture = new THREE.Texture(imageBitmap);
          texture.needsUpdate = true;
          resolve(texture);
        };
        image.onerror = err => {
          console.warn(err);
          reject(err);
        };
        image.crossOrigin = 'Anonymous';
        image.src = url;
      });
    }));
  })());

  const itemletMeshes = [];
  const _dropItemlet = position2D => {
    const geometry = new THREE.PlaneBufferGeometry(dropItemSize, dropItemSize)
      .translate(0, dropItemSize/2, 0);
    const index = Math.floor(Math.random() * itemletTextures.length);
    const texture = itemletTextures[index];
    // const colorPalette = colorPalettes[index];
    const material = new WebaverseShaderMaterial({
      uniforms: {
        cameraBillboardQuaternion: {
          value: new THREE.Quaternion(),
          needsUpdate: false,
        },
        uTex: {
          value: texture,
          needsUpdate: true,
        },
        /* color1: {
          value: new THREE.Color(colorPalette[0]),
          needsUpdate: true,
        },
        color2: {
          value: new THREE.Color(colorPalette[1]),
          needsUpdate: true,
        }, */
      },
      vertexShader: `\
        uniform vec4 cameraBillboardQuaternion;
        varying vec2 vUv;

        vec3 rotate_vertex_position(vec3 position, vec4 q) { 
          return position + 2.0 * cross(q.xyz, cross(q.xyz, position) + q.w * position);
        }

        void main() {
          vec3 pos = position;

          pos = rotate_vertex_position(pos, cameraBillboardQuaternion);

          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
          vUv = uv;
        }
      `,
      fragmentShader: `\
        uniform sampler2D uTex;
        // uniform vec3 color1;
        // uniform vec3 color2;
        varying vec2 vUv;

        void main() {
          vec4 displacementColor = texture2D(uTex, vUv);
          // displacementColor.rgb = mix(color2, color1, vUv.y);
          gl_FragColor = displacementColor;
          if (gl_FragColor.a < 0.1) {
            discard;
          }
        }
      `,
      side: THREE.DoubleSide,
      transparent: true,
    });
    const itemletMesh = new THREE.Mesh(geometry, material);
    itemletMesh.position.set(position2D.x, 0.5, position2D.y);
    itemletMesh.velocity = new THREE.Vector3(-1 + Math.random() * 2, 3, -1 + Math.random() * 2);
    itemletMesh.frustumCulled = false;
    scene.add(itemletMesh);
    itemletMesh.updateMatrixWorld();

    let animation = null;
    itemletMesh.update = (timestamp, timeDiff) => {
      const timeDiffS = timeDiff / 1000;
      const camera = useCamera();
      const localPlayer = useLocalPlayer();

      localEuler.setFromQuaternion(camera.quaternion, 'YXZ');
      localEuler.x = 0;
      localEuler.z = 0;
      itemletMesh.material.uniforms.cameraBillboardQuaternion.value.setFromEuler(localEuler);
      itemletMesh.material.uniforms.cameraBillboardQuaternion.needsUpdate = true;

      if (animation) {
        const timeDiff = timestamp - animation.startTime;
        const factor = timeDiff / animation.duration;
        if (factor < 1) {
          itemletMesh.position.copy(localPlayer.position);
          itemletMesh.position.y += 0.2 + Math.sin(Math.min(factor * 4, 1) * Math.PI) * 0.1;
          itemletMesh.updateMatrixWorld();
        } else {
          scene.remove(itemletMesh);
          itemletMeshes.splice(itemletMeshes.indexOf(itemletMesh), 1);
        }
      } else {
        if (!itemletMesh.velocity.equals(zeroVector)) {
          itemletMesh.position.add(localVector.copy(itemletMesh.velocity).multiplyScalar(timeDiffS));
          itemletMesh.velocity.add(localVector.copy(gravity).multiplyScalar(timeDiffS));
          if (itemletMesh.position.y < floorLimit) {
            itemletMesh.position.y = floorLimit;
            itemletMesh.velocity.set(0, 0, 0);
          }
          itemletMesh.updateMatrixWorld();
        } else {
          const localPosition = localVector.copy(localPlayer.position);
          localPosition.y -= localPlayer.avatar.height;
          // console.log('got pos', localPosition.toArray().join(','), itemletMesh.position.toArray().join(','));

          if (localPosition.distanceTo(itemletMesh.position) < 0.5) {
            animation = {
              startTime: timestamp,
              duration: 1000,
            };
          }
        }
      }
    };

    itemletMeshes.push(itemletMesh);
  };

  useFrame(({timestamp, timeDiff}) => {
    generator.update(timestamp, timeDiff);
    
    for (const itemletMesh of itemletMeshes) {
      itemletMesh.update(timestamp, timeDiff);
    }
  });

  // XXX
  hitManager.addEventListener('hitattempt', e => {
    const {type, args} = e.data;
    if (type === 'sword') {
      const {
        position,
        quaternion,
        // hitHalfHeight,
      } = args;
      for (const mesh of generator.getMeshes()) {
        const hitTarget2D = mesh.hitAttempt(position, quaternion, localVector2D);
        if (hitTarget2D) {
          _dropItemlet(hitTarget2D);

          sounds.playSoundName('bushCut');
        }
      }
    }
  });
  
  return app;
};