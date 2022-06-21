import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import metaversefile from 'metaversefile';
const {useFrame, useCleanup, useApp, useScene, useSound, useMaterials, useInstancing, useRenderer, useCamera, useProcGen, useProcGenManager, useLocalPlayer, useHitManager, useLodder} = metaversefile;

const baseUrl = import.meta.url.replace(/(\/)[^\/\\]*$/, '$1');

const localVector = new THREE.Vector3();
const localVector2 = new THREE.Vector3();
// const localVector3 = new THREE.Vector3();
// const localQuaternion = new THREE.Quaternion();
const localEuler = new THREE.Euler();
const localVector2D = new THREE.Vector2();
// const localVector2D2 = new THREE.Vector2();
// const localVector2D3 = new THREE.Vector2();
const localBox = new THREE.Box3();

const zeroVector = new THREE.Vector3(0, 0, 0);
// const zeroVector2D = new THREE.Vector2(0, 0);
const gravity = new THREE.Vector3(0, -9.8, 0);
const dropItemSize = 0.2;
const chunkWorldSize = 16;
const heightfieldRange = 2;
const heightfieldSizeInChunks = heightfieldRange * 2;
const heightfieldSize = chunkWorldSize * heightfieldSizeInChunks;
const maxAnisotropy = 16;

const height = 0.8;
const radialSegments = 3;
const heightSegments = 8;
const openEnded = false;
const segmentHeight = height / heightSegments;
const numBlades = 4 * 1024;
const cutTime = 1;
const growTime = 60;
const cutGrowTime = cutTime + growTime;
const cutHeightOffset = -1;
const floorLimit = dropItemSize / 2;

const maxInstancesPerDrawCall = numBlades;
const maxDrawCallsPerGeometry = 32;

const windRotation = ((Date.now() / 1000) % 1) * Math.PI * 2;
const heightfieldBase = new THREE.Vector3(-heightfieldSize / 2, 0, -heightfieldSize / 2);
const heightfieldBase2D = new THREE.Vector2(heightfieldBase.x, heightfieldBase.z);

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
/* const _averagePoints = (points, target) => {
  target.copy(points[0]);
  for (let i = 1; i < points.length; i++) {
    target.add(points[i]);
  }
  return target.divideScalar(points.length);
};
const _wrapUvs = (geometry, offset, size) => {
  for (let i = 0; i < geometry.attributes.uv.count; i++) {
    localVector2D.fromArray(geometry.attributes.uv.array, i * 2)
      .multiply(size)
      .add(offset)
      .toArray(geometry.attributes.uv.array, i * 2);
  }
}; */
function mod(a, n) {
  return (a % n + n) % n;
}

//

const fullScreenQuadGeometry = new THREE.PlaneBufferGeometry(2, 2);
/* const fullScreen2xQuadGeometry = (() => {
  // return new THREE.PlaneBufferGeometry(2, 2);
  
  const halfChunkSize = 1 / heightfieldSizeInChunks / 2;
  
  const geometries = [];
  for (let dz = 0; dz < heightfieldSizeInChunks; dz++) {
    for (let dx = 0; dx < heightfieldSizeInChunks; dx++) {
      const uvOffsetX = dx / heightfieldSizeInChunks;
      const uvOffsetZ = dz / heightfieldSizeInChunks;
      const uvOffset = new THREE.Vector2(uvOffsetX, uvOffsetZ);

      const uvSize = new THREE.Vector2().setScalar(1 / heightfieldSizeInChunks);

      for (let dz2 = 0; dz2 < 2; dz2++) {
        for (let dx2 = 0; dx2 < 2; dx2++) {
          const geometryOffsetX = (dx * 2 + dx2) * halfChunkSize;
          const geometryOffsetZ = (dz * 2 + dz2) * halfChunkSize;

          const baseGeometry = new THREE.PlaneBufferGeometry(1, 1)
            .scale(halfChunkSize, halfChunkSize, halfChunkSize)
            .translate(halfChunkSize / 2, halfChunkSize / 2, 0);
          const geometry = baseGeometry.clone()
            .translate(geometryOffsetX, geometryOffsetZ, 0)
            // .scale(0.9, 0.9, 0.9)
          _wrapUvs(geometry, uvOffset, uvSize);
          geometries.push(geometry);
        }
      }
    }
  }

  const geometry = BufferGeometryUtils.mergeBufferGeometries(geometries)
    .translate(-0.5, -0.5, 0)
    .scale(2, 2, 2);
  // window.geometry = geometry;
  return geometry;
})(); */
const fullscreenVertexShader = `\
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0., 1.0);
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
const getAveragePoint = (points, vector) => {
  vector.set(0, 0, 0);
  for (let i = 0; i < points.length; i++) {
    vector.add(points[i]);
  }
  return vector.divideScalar(points.length);
};

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

const silksBaseGeometry = createSilkGrassBladeGeometry();
/* const _makeCutMesh = () => {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(4 * 3), 3));
  geometry.setIndex(new THREE.BufferAttribute(Uint16Array.from([0, 1, 2, 2, 1, 3]), 1));
  const material = new THREE.MeshBasicMaterial({
    color: 0x0000FF,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  return mesh;
}; */
const displacementMapSize = heightfieldSize * 4;
const _makeRenderTarget = () => new THREE.WebGLRenderTarget(displacementMapSize, displacementMapSize, {
  minFilter: THREE.NearestFilter,
  magFilter: THREE.NearestFilter,
  format: THREE.RGBAFormat,
  type: THREE.FloatType,
  wrapS: THREE.ClampToEdgeWrapping,
  wrapT: THREE.ClampToEdgeWrapping,
  stencilBuffer: false,
});
const _getHeightfieldChunk = async (procGenInstance, minX, minZ, lod) => {
  const heightfield = await procGenInstance.dcWorkerManager.getHeightfieldRange(
    minX, minZ,
    chunkWorldSize, chunkWorldSize,
    lod
  );
  return heightfield;
};
const {InstancedBatchedMesh, InstancedGeometryAllocator} = useInstancing();
class SilkGrassMesh extends InstancedBatchedMesh {
  constructor({
    procGenInstance,
  }) {
    const {WebaverseShaderMaterial} = useMaterials();

    const displacementMaps = [
      _makeRenderTarget(),
      _makeRenderTarget(),
    ];
    const allocator = new InstancedGeometryAllocator([
      silksBaseGeometry,
    ], [
      {
        name: 'p',
        Type: Float32Array,
        itemSize: 3,
      },
      {
        name: 'q',
        Type: Float32Array,
        itemSize: 4,
      },
    ], {
      maxInstancesPerDrawCall,
      maxDrawCallsPerGeometry,
      boundingType: 'box',
    });
    const {geometry, textures: attributeTextures} = allocator;
    for (const k in attributeTextures) {
      const texture = attributeTextures[k];
      texture.anisotropy = maxAnisotropy;
    }

    // main material

    const heightfieldMapper = procGenInstance.getHeightfieldMapper();

    // heightfieldRenderTarget.texture.flipY = false;
    // window.heightfieldRenderTarget = heightfieldRenderTarget;

    // debug mesh
    /* {
      const fragmentShader = `\
        uniform sampler2D uHeightfield;
        varying vec2 vUv;
    
        void main() {
          gl_FragColor = texture2D(uHeightfield, vUv);
        }
      `;
      const {WebaverseShaderMaterial} = useMaterials();
      const material = new WebaverseShaderMaterial({
        uniforms: {
          uHeightfield: {
            value: heightfieldRenderTarget.texture,
            needsUpdate: true,
          }
        },
        vertexShader: fullscreenVertexShader,
        fragmentShader,
      });
      const mesh = new THREE.Mesh(fullScreen2xQuadGeometry, material);
      mesh.position.set(0, 1.5, -1.5);
      mesh.frustumCulled = false;
      const scene = useScene();
      scene.add(mesh);
      mesh.updateMatrixWorld();
    } */

    const grassVertexShader = `\
      precision highp float;
      precision highp int;

      uniform float uTime;
      uniform sampler2D uDisplacementMap;
      uniform sampler2D uNoiseTexture;
      uniform sampler2D uSeamlessNoiseTexture;
      uniform sampler2D pTexture;
      uniform sampler2D qTexture;
      uniform float uWindRotation;
      uniform sampler2D uHeightfield;
      uniform sampler2D uHeightfieldFourTap;
      uniform vec2 uHeightfieldBase;
      uniform vec2 uHeightfieldMinPosition;
      // uniform vec2 uHeightfieldPosition;
      uniform float uChunkSize;
      uniform float uHeightfieldSize;
      // uniform float uHeightfieldRange;
      attribute int segment;
      varying vec2 vUv;
      varying vec2 vUvNoise;
      varying vec3 vNormal;
      varying float vTimeDiff;
      varying float vY;
      // varying vec2 vF;
      varying vec3 vNoise;
      // varying vec2 vColor;

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

      vec4 fourTapSample(
        sampler2D atlas,
        vec2 tileUV,
        vec2 tileOffset,
        vec2 tileSize
      ) {
        //Initialize accumulators
        vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
        float totalWeight = 0.0;

        // tileUV.x += 0.5 / uHeightfieldSize;
        // tileUV.y -= 0.5 / uHeightfieldSize;

        // centerUv = mod(centerUv, 0.5) * 0.5;

        // tileUV.y = 1. - tileUV.y;
        // tileOffset.y = 1. - tileOffset.y;

        for (int dx=0; dx<2; ++dx) {
          for (int dy=0; dy<2; ++dy) {
            // tileUV += 0.5;
            // vec2 tileCoord = (tileUV * 0.5 + 0.5 * vec2(dx,dy)) * 2.;
            vec2 tileCoord = 2.0 * mod(0.5 * (tileUV + vec2(dx,dy)), 1.);
            // tileUV -= 0.5;
      
            //Weight sample based on distance to center
            float w = pow(min(abs(1.-tileCoord.x), abs(1.-tileCoord.y)) * 2., 100.);
      
            //Compute atlas coord
            vec2 atlasUV = tileOffset + tileSize * tileCoord;
      
            //Sample and accumulate
            // atlasUV += vec2(0.5, 0.5) / (uHeightfieldSize * 2.0);
            atlasUV.y = 1. - atlasUV.y;
            // atlasUV += vec2(0.5, -0.5) / (uHeightfieldSize * 2.0);
            color += w * texture2D(atlas, atlasUV);
            totalWeight += w;
          }
        }

        /* vF = tileUV.xy;
        if (vF.x >= 0.5) {
          vF.x = 1.;
        } else {
          vF.x = 0.;
        }
        if (vF.y >= 0.5) {
          vF.y = 1.;
        } else {
          vF.y = 0.;
        } */
      
        return color / totalWeight;
        // return vec4(60.);
      }

      float offsetHeight(vec3 offset) {
        vec2 pos2D = offset.xz;

        vec2 posDiff = pos2D - uHeightfieldMinPosition;
        vec2 uvHeightfield = posDiff;
        uvHeightfield /= uHeightfieldSize;
        
        vUv = uvHeightfield;
        vUv.y = 1. - vUv.y;

        vUvNoise = mod(pos2D / uHeightfieldSize, 1.);
        vUvNoise.y = 1. - vUvNoise.y;

        uvHeightfield.x += 0.5 / uHeightfieldSize;
        uvHeightfield.y += 0.5 / uHeightfieldSize;
        uvHeightfield.y = 1. - uvHeightfield.y;
        float heightfieldValue = texture2D(uHeightfieldFourTap, uvHeightfield).r;

        vec2 posDiffMod = uvHeightfield;
        vec2 tileUv = mod(posDiffMod * 4., 1.);
        vec2 tileOffset = floor(mod(posDiffMod, 1.) * 4.) / 4.;
        vec2 tileSize = vec2(1. / (uHeightfieldSize / uChunkSize * 2.));
        
        // float heightfieldValue = fourTapSample(uHeightfieldFourTap, tileUv, tileOffset, tileSize).r;
        // float heightfieldValue = tileOffset.x * 30. + 60.;

        return heightfieldValue;
      }

      const float segmentHeight = ${segmentHeight.toFixed(8)};
      const float heightSegments = ${heightSegments.toFixed(8)};  
      const float topSegmentY = segmentHeight * heightSegments;
      const float cutTime = ${cutTime.toFixed(8)};
      const float growTime = ${growTime.toFixed(8)};
      const float cutGrowTime = ${cutGrowTime.toFixed(8)};
      void main() {
        vec3 pos = position;

        int instanceIndex = gl_DrawID * ${maxInstancesPerDrawCall} + gl_InstanceID;
        const float width = ${attributeTextures.p.image.width.toFixed(8)};
        const float height = ${attributeTextures.p.image.height.toFixed(8)};
        float x = mod(float(instanceIndex), width);
        float y = floor(float(instanceIndex) / width);
        vec2 pUv = (vec2(x, y) + 0.5) / vec2(width, height);
        vec3 p = texture2D(pTexture, pUv).xyz;
        vec4 q = texture2D(qTexture, pUv).xyzw;

        float h = offsetHeight(p);

        // time diff
        vec4 displacementColor = texture2D(uDisplacementMap, vUv);
        float cutStartTime = displacementColor.w;
        vTimeDiff = uTime - cutStartTime;

        // compute cut/grow variables
        float segmentStartY = float(segment) * segmentHeight;
        float cutY = 0.1;
        float cutSegmentY = floor(cutY / segmentHeight) * segmentHeight;
        bool isCuttableY = (cutY > 0. && cutY < segmentStartY);
        bool isCut = isCuttableY && (vTimeDiff < cutTime);
        bool isGrow = isCuttableY && (cutStartTime > 0.) && (vTimeDiff >= cutTime && vTimeDiff < cutGrowTime);

        // grow
        if (isGrow) {
          vec3 bottomOfCutBlade = vec3(0., cutSegmentY /* + segmentHeight */, 0.);
          float scaleFactor = min((vTimeDiff - cutTime) / growTime, 1.);

          // grow
          pos -= bottomOfCutBlade;
          pos.y *= scaleFactor;
          pos += bottomOfCutBlade;
        }

        // noise
        vY = pos.y;
        vNoise = texture2D(uNoiseTexture, vUvNoise).xyz;

        // cut
        if (isCut) {
          vec3 centerOfBlade = vec3(0., (cutSegmentY + topSegmentY) * 0.5, 0.);
          // float scaleFactor = max(1. - vTimeDiff / cutTime, 0.);

          // acceleration / velocity
          const float GRAVITY = -9.8;
          const float CUT_VELOCITY = 2.;

          vec2 vUvNoise2 = mod(vUvNoise * 3., 1.); // to not conflict with rotation axis
          vec2 directionXZ = -1. + texture2D(uNoiseTexture, vUvNoise).xz * 2.;
          
          // compute the acceleration / velocity offset
          vec3 gravityVector = vec3(0., GRAVITY * vTimeDiff * vTimeDiff * 0.5, 0.);
          float cutSpeed = texture2D(uNoiseTexture, vUvNoise).w * 3.;
          vec3 direction = vec3(directionXZ.x, CUT_VELOCITY, directionXZ.y);
          vec3 velocityVector = direction * vTimeDiff * cutSpeed;

          float centerOfBladePositionY = centerOfBlade.y + velocityVector.y + gravityVector.y;
          if (centerOfBladePositionY >= 0.) {
            // scale + rotation
            pos -= centerOfBlade;
            vec3 rotationAxis = normalize(-1. + texture2D(uNoiseTexture, vUvNoise).xyz * 2.);
            vec4 q = quat_from_axis_angle(rotationAxis, uTime * 2. * PI * 0.2);
            pos = rotate_vertex_position(pos, q);
            pos += centerOfBlade;

            // velocity + position
            pos += velocityVector + gravityVector;
          } else {
            pos = vec3(0.);
          }
        }

        // instance offset
        {
          pos = rotate_vertex_position(pos, q);
          pos.xz += p.xz;
        }

        // wind
        if (!isCut) {
          float windOffsetX = (texture2D(
            uSeamlessNoiseTexture,
            vUvNoise + vec2(uTime * 0.05, 0.)
          ).r - 0.5) * 2. * pos.y;
          float windOffsetY = 0.;
          float windOffsetZ = 0.;
          vec3 windOffset = vec3(windOffsetX, windOffsetY, windOffsetZ);
          vec4 q2 = quat_from_axis_angle(vec3(0., 1., 0.), uWindRotation);
          windOffset = rotate_vertex_position(windOffset, q2);
          pos += windOffset * 1.;
        }

        // displacement bend
        if (!isCut) {
          vec4 displacement = texture2D(uDisplacementMap, vUv);
          pos.xz += displacement.xy * pow(pos.y, 0.5) * 0.5;
        }

        // height offset
        pos.y += h;

        // output
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        
        vNormal = normal;
      }
    `;
    const grassFragmentShader = `\
      precision highp float;
      precision highp int;

      #define PI 3.1415926535897932384626433832795

      uniform float uTime;
      uniform sampler2D uDisplacementMap;
      varying float vOffset;
      varying vec2 vUv;
      varying vec3 vNormal;
      varying float vTimeDiff;
      varying float vY;
      // varying vec2 vF;
      varying vec3 vNoise;
      // varying vec2 vColor;
      // varying vec2 vUvNoise;

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
        vec4 displacementColor = texture2D(uDisplacementMap, vUv);
        
        gl_FragColor.rgb = color *
          (0.4 + rand(floor(100. + (vNoise.x + vNoise.y + vNoise.z) * 15.)) * 0.6) *
          (0.2 + vY/height * 0.8);
        // gl_FragColor.rgb = vec3(0.);
        // gl_FragColor.rb = vUv;
        // gl_FragColor.rgb = displacementColor.rgb;
        // gl_FragColor.rgb = vec3(vF.x, 0., vF.y);
        // gl_FragColor.rb = vUvNoise;
        // gl_FragColor.g = 0.;
        gl_FragColor.a = 1.;
      }
    `;
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
        pTexture: {
          value: attributeTextures['p'],
          needsUpdate: true,
        },
        qTexture: {
          value: attributeTextures['q'],
          needsUpdate: true,
        },
        uHeightfield: {
          value: heightfieldMapper.heightfieldRenderTarget.texture,
          needsUpdate: true,
        },
        uHeightfieldFourTap: {
          value: heightfieldMapper.heightfieldFourTapRenderTarget.texture,
          needsUpdate: true,
        },
        uHeightfieldBase: {
          value: heightfieldBase2D,
          needsUpdate: true,
        },
        /* uHeightfieldPosition: {
          value: new THREE.Vector2(),
          needsUpdate: true,
        }, */
        uHeightfieldMinPosition: {
          value: new THREE.Vector2(),
          needsUpdate: true,
        },
        uChunkSize: {
          value: chunkWorldSize,
          needsUpdate: true,
        },
        uHeightfieldSize: {
          value: heightfieldMapper.terrainSize,
          needsUpdate: true,
        },
        /* uHeightfieldRange: {
          value: heightfieldRange,
          needsUpdate: true,
        }, */
      },
      vertexShader: grassVertexShader,
      fragmentShader: grassFragmentShader,
      // transparent: true,
    });
    super(geometry, material, allocator);
    this.frustumCulled = false;

    // local members
    this.procGenInstance = procGenInstance;
    this.allocator = allocator;
    this.displacementMaps = displacementMaps;
    this.heightfieldMapper = heightfieldMapper;

    // update functions

    this.updateCoord = (min2xCoord) => {
      const delta = this.heightfieldMapper.updateCoord(min2xCoord, localVector2D);
      
      material.uniforms.uHeightfieldMinPosition.value.copy(this.heightfieldMapper.heightfieldMinPosition);
      material.uniforms.uHeightfieldMinPosition.needsUpdate = true;
    
      // displacement animation
      displacementAnimationScene.mesh.material.uniforms.uHeightfieldMinPosition.value
        .copy(this.heightfieldMapper.heightfieldMinPosition);
      displacementAnimationScene.mesh.material.uniforms.uHeightfieldMinPosition.needsUpdate = true;

      // cut
      cutScene.mesh.material.uniforms.uHeightfieldMinPosition.value
        .copy(this.heightfieldMapper.heightfieldMinPosition);
      cutScene.mesh.material.uniforms.uHeightfieldMinPosition.needsUpdate = true;

      // copy displacement scene
      this.renderDisplacementMapDelta(delta);
    };
    const displacementAnimationScene = (() => {
      const heightfieldPlaneGeometry = new THREE.PlaneBufferGeometry(1, 1)
        .rotateX(-Math.PI / 2)
        .translate(0.5, 0, 0.5)
        .scale(heightfieldMapper.terrainSize, 1, heightfieldMapper.terrainSize);
      const displacementVertexShader = `\
        uniform vec2 uHeightfieldMinPosition;
        uniform float uHeightfieldSize;
        varying vec2 vUv;
        varying vec3 vPosition;

        void main() {
          vUv = uv;
          vec4 position2 = modelMatrix * vec4(position, 1.0);
          vPosition = position2.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `;
      const fullscreenFragmentShader = `\
        uniform vec3 uPlayerPosition;
        // uniform vec3 uWorldPosition;
        uniform sampler2D uNoiseTexture;
        uniform sampler2D uDisplacementMap;
        uniform vec2 uHeightfieldMinPosition;
        uniform float uHeightfieldSize;
        uniform sampler2D uHeightfieldFourTap;
        uniform vec3 pA1;
        uniform vec3 pA2;
        uniform vec3 pB1;
        uniform vec3 pB2;
        varying vec2 vUv;
        varying vec3 vPosition;
        // const float chunkWorldSize = ${chunkWorldSize.toFixed(8)};
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
          vec2 pos2D = vPosition.xz;
          
          vec2 posDiff = pos2D - uHeightfieldMinPosition;
          vec2 uvHeightfield = posDiff;
          uvHeightfield /= uHeightfieldSize;
          
          uvHeightfield.x += 0.5 / uHeightfieldSize;
          uvHeightfield.y += 0.5 / uHeightfieldSize;
          uvHeightfield.y = 1. - uvHeightfield.y;
          float heightfieldValue = texture2D(uHeightfieldFourTap, uvHeightfield).r;

          //

          vec3 virtual = vPosition;
          virtual.y += heightfieldValue;

          // virtual.y = 0.;
          // vec3 uPlayerPosition2 = uPlayerPosition;
          // uPlayerPosition2.y = 0.;
          
          float distanceToPlayerVertical = distanceToLine(
            virtual,
            uPlayerPosition,
            uPlayerPosition + vec3(0., -1.5, 0.) // XXX should use the real avatar height
            // uPlayerPosition2
          );

          vec2 direction = distanceToPlayerVertical > 0.0 ? normalize(virtual.xz - uPlayerPosition.xz) : vec2(0.0, 0.0);

          vec4 oldColor = texture2D(uDisplacementMap, vUv);

          vec4 newColor = vec4(direction, oldColor.zw);
          float distanceFactor = min(max(maxDistance - distanceToPlayerVertical, 0.), 1.);
          
          float localLearningRate = learningRate;
          if (distanceFactor > 0.0) {
            localLearningRate = 1.;
          }
          gl_FragColor = vec4(
            min(max(
              oldColor.xy * (1. - learningRate) +
                (newColor.xy * distanceFactor) * localLearningRate,
            vec2(-1.)), vec2(1.)),
            -1.,
            newColor.w
          );
          // gl_FragColor = vec4(vUv.x, 0., vUv.y, 1.);
          // gl_FragColor = vec4(heightfieldValue, 0., 0., 1.);
        }
      `;
      const fullscreenMaterial = new THREE.ShaderMaterial({
        uniforms: {
          uPlayerPosition: {
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
          uHeightfieldSize: {
            value: heightfieldMapper.terrainSize,
            needsUpdate: true,
          },
          uHeightfieldMinPosition: {
            value: new THREE.Vector2(),
            needsUpdate: true,
          },
          uHeightfieldFourTap: {
            value: heightfieldMapper.heightfieldFourTapRenderTarget.texture,
            needsUpdate: true,
          },
        },
        vertexShader: displacementVertexShader,
        fragmentShader: fullscreenFragmentShader,
        // side: THREE.DoubleSide,
      });
      const displacementAnimationMesh = new THREE.Mesh(heightfieldPlaneGeometry, fullscreenMaterial);
      displacementAnimationMesh.frustumCulled = false;
      const scene = new THREE.Scene();
      scene.add(displacementAnimationMesh);
      scene.mesh = displacementAnimationMesh;

      scene.camera = new THREE.OrthographicCamera(
        0, heightfieldMapper.terrainSize,
        0, -heightfieldMapper.terrainSize,
        -1000, 1000
      );
      scene.camera.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
      scene.camera.updateMatrixWorld();
      
      scene.update = (displacementMapTexture) => {
        const localPlayer = useLocalPlayer();
        fullscreenMaterial.uniforms.uPlayerPosition.value.copy(localPlayer.position);
        fullscreenMaterial.uniforms.uPlayerPosition.needsUpdate = true;

        fullscreenMaterial.uniforms.uDisplacementMap.value = displacementMapTexture;
        fullscreenMaterial.uniforms.uDisplacementMap.needsUpdate = true;

        // uHeightfieldMinPosition is updated when the coord changes
        displacementAnimationMesh.position.setFromMatrixPosition(this.matrixWorld)
          .add(localVector.set(
            fullscreenMaterial.uniforms.uHeightfieldMinPosition.value.x,
            0,
            fullscreenMaterial.uniforms.uHeightfieldMinPosition.value.y
          ));
        displacementAnimationMesh.updateMatrixWorld();
        scene.camera.position.copy(displacementAnimationMesh.position);
        scene.camera.updateMatrixWorld();
      };
      return scene;
    })();
    const displacementCopyScene = (() => {
      const fullscreenFragmentShader = `\
        uniform vec2 uPositionDelta;
        uniform float uHeightfieldSize;
        uniform sampler2D uDisplacementMap;
        varying vec2 vUv;

        void main() {
          vec2 pos2D = vUv;
          vec2 pd = uPositionDelta;
          pd /= uHeightfieldSize;
          // pd.y = 1. - pd.y;
          pd.y *= -1.;
          vec2 uv = pos2D + pd;
          // uv.y = 1. - uv.y;
          // uv = mod(uv, 1.);
          if (uv.x >= 0. && uv.x <= 1. && uv.y >= 0. && uv.y <= 1.) {
            gl_FragColor = texture2D(uDisplacementMap, uv);
          } else {
            gl_FragColor = vec4(0., 0., 0., 0.);
          }
        }
      `;
      const fourTapFullscreenMaterial = new THREE.ShaderMaterial({
        uniforms: {
          uHeightfieldSize: {
            value: heightfieldMapper.terrainSize,
            needsUpdate: true,
          },
          uPositionDelta: {
            value: new THREE.Vector2(),
            needsUpdate: true,
          },
          uDisplacementMap: {
            value: displacementMaps[0].texture,
            needsUpdate: false,
          },
        },
        vertexShader: fullscreenVertexShader,
        fragmentShader: fullscreenFragmentShader,
      });
      const fourTapQuadMesh = new THREE.Mesh(fullScreenQuadGeometry, fourTapFullscreenMaterial);
      fourTapQuadMesh.frustumCulled = false;
      const scene = new THREE.Scene();
      scene.add(fourTapQuadMesh);
      // scene.mesh = fourTapQuadMesh;
      scene.update = (positionDelta, displacementMapTexture) => {
        fourTapFullscreenMaterial.uniforms.uPositionDelta.value.copy(positionDelta);
        fourTapFullscreenMaterial.uniforms.uPositionDelta.needsUpdate = true;

        fourTapFullscreenMaterial.uniforms.uDisplacementMap.value = displacementMapTexture;
        fourTapFullscreenMaterial.uniforms.uDisplacementMap.needsUpdate = true;
      };
      return scene;
    })();
    const cutScene = (() => {
      const cutGeometry = new THREE.PlaneBufferGeometry(1, 1)
        .rotateX(-Math.PI / 2)
        .translate(0.5, 0, 0.5)
        .scale(heightfieldMapper.terrainSize, 1, heightfieldMapper.terrainSize);
      const cutVertexShader = `\
        // uniform vec2 uHeightfieldMinPosition;
        // uniform float uHeightfieldSize;
        varying vec2 vUv;
        varying vec3 vPosition;

        void main() {
          vUv = uv;
          vec4 position2 = modelMatrix * vec4(position, 1.0);
          vPosition = position2.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `;
      const fullscreenFragmentShader2 = `\
        uniform vec3 uWorldPosition;
        uniform sampler2D uDisplacementMap;
        uniform sampler2D uHeightfieldFourTap;
        uniform float uHeightfieldSize;
        uniform vec2 uHeightfieldMinPosition;
        uniform float uTime;
        uniform float uHitStamp;
        uniform vec3 pA1;
        uniform vec3 pA2;
        uniform vec3 pB1;
        uniform vec3 pB2;
        varying vec2 vUv;
        varying vec3 vPosition;

        const float chunkWorldSize = ${chunkWorldSize.toFixed(8)};
        const float learningRate = 0.005;
        const float maxDistance = 0.6;
        const float cutYRange = 1.5;

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
        const float heightSegments = ${heightSegments.toFixed(8)};
        void main() {
          vec2 pos2D = vPosition.xz;
          
          vec2 posDiff = pos2D - uHeightfieldMinPosition;
          vec2 uvHeightfield = posDiff;
          uvHeightfield /= uHeightfieldSize;
          
          uvHeightfield.x += 0.5 / uHeightfieldSize;
          uvHeightfield.y += 0.5 / uHeightfieldSize;
          uvHeightfield.y = 1. - uvHeightfield.y;
          float heightfieldValue = texture2D(uHeightfieldFourTap, uvHeightfield).r;

          //

          vec3 virtual = vPosition;
          virtual.y += heightfieldValue;

          // vec2 virtualXZ = vec2(vUv.x, vUv.y) * chunkWorldSize;
          // virtualXZ += uWorldPosition.xz;

          vec4 color = texture2D(uDisplacementMap, vUv);

          vec2 a = pA1.xz;
          vec2 b = pA2.xz;
          vec2 c = pB1.xz;
          vec2 d = pB2.xz;
          float cutHeight = (pA1.y + pA2.y + pB1.y + pB2.y) / 4.;
          float timeDiff = uTime - color.w;
          if (
            (
              isPointInTriangle(virtual.xz, a, b, c) || isPointInTriangle(virtual.xz, b, d, c)
            ) &&
            abs(virtual.y - cutHeight) < cutYRange &&
            timeDiff > (cutTime + growTime * 0.5)
          ) {
            // color.z = (pA1.y + pA2.y + pB1.y + pB2.y) / 4.;
            color.z = uHitStamp;
            color.w = uTime;
          }
          gl_FragColor = color;
        }
      `;
      const fullscreenMaterial2 = new THREE.ShaderMaterial({
        uniforms: {
          /* uPlayerPosition: {
            value: new THREE.Vector3(0, 0, 0),
            needsUpdate: false,
          }, */
          uWorldPosition: {
            value: new THREE.Vector3(0, 0, 0),
            needsUpdate: true,
          },
          uDisplacementMap: {
            value: displacementMaps[0].texture,
            needsUpdate: true,
          },
          uNoiseTexture: {
            value: getNoiseTexture(),
            needsUpdate: true,
          },
          uTime: {
            value: 0,
            needsUpdate: true,
          },
          uHitStamp: {
            value: 0,
            needsUpdate: true,
          },
          pA1: {
            value: new THREE.Vector3(),
            needsUpdate: true,
          },
          pA2: {
            value: new THREE.Vector3(),
            needsUpdate: true,
          },
          pB1: {
            value: new THREE.Vector3(),
            needsUpdate: true,
          },
          pB2: {
            value: new THREE.Vector3(),
            needsUpdate: true,
          },
          uHeightfieldMinPosition: {
            value: new THREE.Vector2(),
            needsUpdate: true,
          },
          uHeightfield: {
            value: heightfieldMapper.heightfieldRenderTarget.texture,
            needsUpdate: true,
          },
          uHeightfieldFourTap: {
            value: heightfieldMapper.heightfieldFourTapRenderTarget.texture,
            needsUpdate: true,
          },
          uHeightfieldSize: {
            value: heightfieldMapper.terrainSize,
            needsUpdate: true,
          },
        },
        vertexShader: cutVertexShader,
        fragmentShader: fullscreenFragmentShader2,
        // side: THREE.DoubleSide,
      });
      const fullscreenQuadMesh2 = new THREE.Mesh(cutGeometry, fullscreenMaterial2);
      fullscreenQuadMesh2.frustumCulled = false;
      const scene2 = new THREE.Scene();
      scene2.add(fullscreenQuadMesh2);
      scene2.mesh = fullscreenQuadMesh2;
      
      scene2.camera = new THREE.OrthographicCamera(
        0, heightfieldMapper.terrainSize,
        0, -heightfieldMapper.terrainSize,
        -1000, 1000
      );
      scene2.camera.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
      scene2.camera.updateMatrixWorld();

      scene2.update = (pA1, pA2, pB1, pB2, hitStamp, timestampS, displacementMapTexture) => {
        // fullscreenMaterial2.uniforms.uWorldPosition.value.setFromMatrixPosition(mesh.matrixWorld);
        // fullscreenMaterial2.uniforms.uWorldPosition.needsUpdate = true;

        // const localPlayer = useLocalPlayer();
        // fullscreenMaterial2.uniforms.uPlayerPosition.value.copy(localPlayer.position);
        // fullscreenMaterial2.uniforms.uPlayerPosition.needsUpdate = true;

        fullscreenMaterial2.uniforms.uDisplacementMap.value = displacementMapTexture;
        fullscreenMaterial2.uniforms.uDisplacementMap.needsUpdate = true;

        fullscreenMaterial2.uniforms.pA1.value.copy(pA1);
        fullscreenMaterial2.uniforms.pA1.needsUpdate = true;
        fullscreenMaterial2.uniforms.pA2.value.copy(pA2);
        fullscreenMaterial2.uniforms.pA2.needsUpdate = true;
        fullscreenMaterial2.uniforms.pB1.value.copy(pB1);
        fullscreenMaterial2.uniforms.pB1.needsUpdate = true;
        fullscreenMaterial2.uniforms.pB2.value.copy(pB2);
        fullscreenMaterial2.uniforms.pB2.needsUpdate = true;

        fullscreenMaterial2.uniforms.uHitStamp.value = hitStamp;
        fullscreenMaterial2.uniforms.uHitStamp.needsUpdate = true;

        fullscreenMaterial2.uniforms.uTime.value = timestampS;
        fullscreenMaterial2.uniforms.uTime.needsUpdate = true;

        //

        fullscreenQuadMesh2.position.setFromMatrixPosition(this.matrixWorld)
          .add(localVector.set(
            fullscreenMaterial2.uniforms.uHeightfieldMinPosition.value.x,
            0,
            fullscreenMaterial2.uniforms.uHeightfieldMinPosition.value.y
          ));
        fullscreenQuadMesh2.updateMatrixWorld();
        scene2.camera.position.copy(fullscreenQuadMesh2.position);
        scene2.camera.updateMatrixWorld();
      };
      return scene2;
    })();
    this.renderAnimation = () => {
      const renderer = useRenderer();
      const context = renderer.getContext();
      // const camera = useCamera();

      {
        // update
        displacementAnimationScene.update(displacementMaps[0].texture);

        // push state
        const oldRenderTarget = renderer.getRenderTarget();
        context.disable(context.SAMPLE_ALPHA_TO_COVERAGE);

        // render
        renderer.setRenderTarget(displacementMaps[1]);
        renderer.clear();
        renderer.render(displacementAnimationScene, displacementAnimationScene.camera);

        // pop state
        renderer.setRenderTarget(oldRenderTarget);
        context.enable(context.SAMPLE_ALPHA_TO_COVERAGE);
      }

      this.flipRenderTargets();
    };
    this.renderDisplacementMapDelta = delta => {
      const renderer = useRenderer();
      const context = renderer.getContext();
      const camera = useCamera();

      {
        // update
        displacementCopyScene.update(delta, displacementMaps[0].texture);

        // push state
        const oldRenderTarget = renderer.getRenderTarget();
        context.disable(context.SAMPLE_ALPHA_TO_COVERAGE);

        // render
        renderer.setRenderTarget(displacementMaps[1]);
        renderer.clear();
        renderer.render(displacementCopyScene, camera);

        // pop state
        renderer.setRenderTarget(oldRenderTarget);
        context.enable(context.SAMPLE_ALPHA_TO_COVERAGE);
      }
      
      this.flipRenderTargets();
    };
    this.renderCut = (pA1, pA2, pB1, pB2, hitStamp, timestampS, displacementMapTexture) => {
      const renderer = useRenderer();
      const context = renderer.getContext();
      // const camera = useCamera();

      {
        // update
        cutScene.update(pA1, pA2, pB1, pB2, hitStamp, timestampS, displacementMapTexture);
        
        // push state
        const oldRenderTarget = renderer.getRenderTarget();
        context.disable(context.SAMPLE_ALPHA_TO_COVERAGE);

        // render
        renderer.setRenderTarget(displacementMaps[1]);
        renderer.clear();
        renderer.render(cutScene, cutScene.camera);

        // pop state
        renderer.setRenderTarget(oldRenderTarget);
        context.enable(context.SAMPLE_ALPHA_TO_COVERAGE);
      }

      this.flipRenderTargets();
    };
    this.flipRenderTargets = () => {
      const temp = displacementMaps[0];
      displacementMaps[0] = displacementMaps[1];
      displacementMaps[1] = temp;
    };

    // this.lastHitTime = -Infinity;
    this.hitStamps = 0;

    /* // XXX debugging
    const heightfieldMesh = (() => {
      const geometry = new THREE.PlaneBufferGeometry(1, 1);
      const material = new THREE.MeshBasicMaterial({
        map: heightfieldRenderTarget.texture,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.frustumCulled = false;
      return mesh;
    })();
    const scene = useScene();
    scene.add(heightfieldMesh);
    heightfieldMesh.updateMatrixWorld();
    this.heightfieldMesh = heightfieldMesh; */
  }
  async addChunk(chunk, {
    signal,
  } = {}) {
    if (chunk.y === 0) {
      let live = true;
      signal.addEventListener('abort', e => {
        live = false;
      });

      await Promise.all([
        (async () => {
          const _getGrassData = async () => {
            const lod = 1;
            const result = await this.procGenInstance.dcWorkerManager.createGrassSplat(chunk.x * chunkWorldSize, chunk.z * chunkWorldSize, lod);
            return result;
          };
          const result = await _getGrassData();
          if (!live) return;
    
          const _renderSilksGeometry = (drawCall, ps, qs) => {
            const pTexture = drawCall.getTexture('p');
            const pOffset = drawCall.getTextureOffset('p');
            const qTexture = drawCall.getTexture('q');
            const qOffset = drawCall.getTextureOffset('q');
    
            pTexture.image.data.set(ps, pOffset);
            qTexture.image.data.set(qs, qOffset);
    
            drawCall.updateTexture('p', pOffset, ps.length);
            drawCall.updateTexture('q', qOffset, qs.length);
    
            drawCall.setInstanceCount(ps.length / 3);
          };
    
          localBox.setFromCenterAndSize(
            localVector.set(
              (chunk.x + 0.5) * chunkWorldSize,
              (chunk.y + 0.5) * chunkWorldSize,
              (chunk.z + 0.5) * chunkWorldSize
            ),
            localVector2.set(chunkWorldSize, chunkWorldSize * 10, chunkWorldSize)
          );
          const drawCall = this.allocator.allocDrawCall(0, localBox);
          _renderSilksGeometry(drawCall, result.ps, result.qs);
    
          signal.addEventListener('abort', e => {
            this.allocator.freeDrawCall(drawCall);
          });
        })(),
        (async () => {
          const lod = 1;
          const heightfield = await _getHeightfieldChunk(this.procGenInstance, chunk.x * chunkWorldSize, chunk.z * chunkWorldSize, lod);
          if (!live) return;
          // console.log('got heightfield', chunk.x, chunk.z, heightfield);

          // chunkDataTexture.image.data.set(heightfield);

          // const renderer = useRenderer();
          // const heightfieldMin = this.material.uniforms.uHeightfieldPosition.value;

          const _getWorldModPosition = target => {
            target.set(chunk.x * chunkWorldSize, 0, chunk.z * chunkWorldSize)
              .sub(heightfieldBase);
            target.x = mod(target.x, this.heightfieldMapper.terrainSize);
            target.z = mod(target.z, this.heightfieldMapper.terrainSize);
            return target;
          };
          const position = _getWorldModPosition(localVector);

          // console.log('render update', position.x, position.y);

          this.heightfieldMapper.renderHeightfieldUpdate(position, heightfield);
          this.heightfieldMapper.updateFourTapHeightfield();
          
          /* if (position.x < 0 || position.y < 0) {
            debugger;
          }
          // console.log('copy position', position.x, position.y);
          renderer.copyTextureToTexture(position, chunkDataTexture, this.material.uniforms.uHeightfield.value); */

          signal.addEventListener('abort', e => {
            const position = _getWorldModPosition(localVector);
            this.heightfieldMapper.clearHeightfieldChunk(position);
            this.heightfieldMapper.updateFourTapHeightfield();
            /* const renderer = useRenderer();
            const heightfieldMin = this.material.uniforms.uHeightfieldPosition.value;
            const chunkPosition = localVector2D2.set(chunk.x * chunkWorldSize, chunk.z * chunkWorldSize);
            const position = localVector2D3.copy(chunkPosition).sub(heightfieldMin);
            renderer.copyTextureToTexture(position, blankChunkDataTexture, this.material.uniforms.uHeightfield.value); */
          });
        })(),
      ]);
    }
  }
  update(timestamp, timeDiff) {
    this.renderAnimation();

    // update material
    const timestampS = timestamp / 1000;
    this.material.uniforms.uTime.value = timestampS;
    this.material.uniforms.uTime.needsUpdate = true;

    this.material.uniforms.uDisplacementMap.value = this.displacementMaps[0].texture;
    this.material.uniforms.uDisplacementMap.needsUpdate = true;

    for (const itemletMesh of this.itemletMeshes) {
      itemletMesh.update(timestamp, timeDiff);
    }
  }
  itemletMeshes = [];
  async checkCut(hitStamp) {
    const renderer = useRenderer();
    const gl = renderer.getContext();

    const texture = this.displacementMaps[0].texture;
    const webglTexture = renderer.textures.getTexture(texture);
    
    const x = 0;
    const y = 0;
    const w = displacementMapSize;
    const h = displacementMapSize;
    const format = gl.RGBA;
    const type = gl.FLOAT;
    const outputBuffer = new Float32Array(displacementMapSize * displacementMapSize * 4);
    await renderer.readPixelsAsync(x, y, w, h, format, type, webglTexture, outputBuffer);

    const numCuts = (() => {
      let numCuts = 0;
      for (let i = 0; i < outputBuffer.length; i += 4) {
        const currentHitStamp = outputBuffer[i + 2];
        if (currentHitStamp === hitStamp) {
          numCuts++;
        }
      }
      return numCuts;
    })();
    return numCuts;
  }
  hitAttempt(position, quaternion) {
    const scene = useScene();
    const {WebaverseShaderMaterial} = useMaterials();
    const sounds = useSound();

    const timestamp = performance.now();

    const _dropItemlet = position => {
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
      itemletMesh.position.copy(position)
        .add(localVector.setY(0, 0.5, 0));
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
            this.itemletMeshes.splice(this.itemletMeshes.indexOf(itemletMesh), 1);
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

            if (localPosition.distanceTo(itemletMesh.position) < 0.5) {
              animation = {
                startTime: timestamp,
                duration: 1000,
              };
            }
          }
        }
      };

      this.itemletMeshes.push(itemletMesh);
    };

    const pointA1 = position.clone()
      .add(new THREE.Vector3(-1.5, cutHeightOffset, 0.1).applyQuaternion(quaternion));
    const pointA2 = position.clone()
      .add(new THREE.Vector3(-0.7, cutHeightOffset, -1.5).applyQuaternion(quaternion));
    const pointB1 = position.clone()
      .add(new THREE.Vector3(1.5, cutHeightOffset, 0.1).applyQuaternion(quaternion));
    const pointB2 = position.clone()
      .add(new THREE.Vector3(0.7, cutHeightOffset, -1.5).applyQuaternion(quaternion));
    /* [pointA1, pointA2, pointB1, pointB2].forEach((point, i) => {
      point.toArray(cutMesh.geometry.attributes.position.array, i * 3);
    });
    cutMesh.geometry.attributes.position.needsUpdate = true; */

    const hitStamp = ++this.hitStamps;
    const timestampS = timestamp / 1000;
    this.renderCut(pointA1, pointA2, pointB1, pointB2, hitStamp, timestampS, this.displacementMaps[0].texture);

    // check if we cut anything
    (async () => {
      const numCuts = await this.checkCut(hitStamp);
      const cut = numCuts >= 8;
      if (cut) {
        sounds.playSoundName('bushCut');

        const pointAverage = getAveragePoint([pointA1, pointA2, pointB1, pointB2], new THREE.Vector3());
        _dropItemlet(pointAverage);
      }
    })();
  }
};

class GrassChunkGenerator {
  constructor({
    procGenInstance,
  }) {
    this.mesh = new SilkGrassMesh({
      procGenInstance,
    });
  }
  getChunks() {
    return this.mesh;
  }
  generateChunk(chunk) {
    const abortController = new AbortController();
    const {signal} = abortController;
    
    (async () => {
      this.mesh.addChunk(chunk, {
        signal,
      });
    })();    

    chunk.binding = {
      abortController,
    };
  }
  disposeChunk(chunk) {
    const {abortController} = chunk.binding;
    abortController.abort();
    chunk.binding = null;
  }
  hitAttempt(position, quaternion, target2D) {
    return this.mesh.hitAttempt(position, quaternion, target2D);
  }
  update(timestamp, timeDiff) {
    this.mesh.update(timestamp, timeDiff);
  }
  destroy() {
    // nothing; the owning lod tracker disposes of our contents
  }
}

let itemletTextures = null;

export default e => {
  const app = useApp();
  const hitManager = useHitManager();
  const procGenManager = useProcGenManager();

  app.name = 'silk-grass';

  const seed = app.getComponent('seed') ?? null;
  let range = app.getComponent('range') ?? null;
  const wait = app.getComponent('wait') ?? false;
  if (range) {
    range = new THREE.Box3(
      new THREE.Vector3(range[0][0], range[0][1], range[0][2]),
      new THREE.Vector3(range[1][0], range[1][1], range[1][2])
    );
  }

  const procGenInstance = procGenManager.getInstance(seed, range);

  const generator = new GrassChunkGenerator({
    procGenInstance,
  });
  /* const tracker = new LodChunkTracker(generator, {
    chunkWorldSize,
    numLods,
  }); */
  const numLods = 1;
  const tracker = procGenInstance.getChunkTracker({
    numLods,
    // trackY: true,
    // relod: true,
  });
  const chunkadd = e => {
    const {chunk} = e.data;
    generator.generateChunk(chunk);
  };
  tracker.addEventListener('chunkadd', chunkadd);
  const chunkremove = e => {
    const {chunk} = e.data;
    generator.disposeChunk(chunk);
  };
  tracker.addEventListener('chunkremove', chunkremove);

  const chunksMesh = generator.getChunks();
  app.add(chunksMesh);
  chunksMesh.updateMatrixWorld();

  const coordupdate = e => {
    const {coord} = e.data;
    chunksMesh.updateCoord(coord);
  };
  tracker.addEventListener('coordupdate', coordupdate);

  useFrame(({timestamp, timeDiff}) => {
    if (!range) {
      const localPlayer = useLocalPlayer();
      tracker.update(localPlayer.position);
    }
    generator.update(timestamp, timeDiff);
  });

  useCleanup(() => {
    tracker.destroy();
    // tracker.removeEventListener('coordupdate', coordupdate);
  });

  // itemlets support
  // XXX this should be a type of drop in the drop manager
  e.waitUntil((async () => {
    await Promise.all([
      (async () => {
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
      })(),
      (async () => {
        if (wait) {
          await new Promise((accept, reject) => {
            tracker.addEventListener('update', () => {
              accept();
            }, {
              once: true,
            });
          });
        }
      })(),
    ]);
  })());

  // XXX
  hitManager.addEventListener('hitattempt', e => {
    const {type, args} = e.data;
    if (type === 'sword') {
      const {
        position,
        quaternion,
        // hitHalfHeight,
      } = args;
      generator.hitAttempt(position, quaternion);
    }
  });
  
  return app;
};