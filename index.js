import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import metaversefile from 'metaversefile';
const {useFrame, useScene, useMaterials, useRenderer, useCamera, useProcGen, useLocalPlayer, useHitManager} = metaversefile;

const localVector = new THREE.Vector3();
const localQuaternion = new THREE.Quaternion();

const upVector = new THREE.Vector3(0, 1, 0);

const radiusTop = 0.01;
const radiusBottom = radiusTop;
const height = 0.8;
const radialSegments = 4;
const heightSegments = 8;
const openEnded = false;
const segmentHeight = height / heightSegments;
const numBlades = 8 * 1024;
const range = 5;
const cutTime = 1;
const growTime = 1;
const cutGrowTime = cutTime + growTime;

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
  const texture = new THREE.Texture(canvas);
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

const createSilkGrassBladeGeometry = () => {
  const geometryNonInstanced = (() => {
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
    return result;
  })();
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
const _makeSilksMesh = () => {
  const {WebaverseShaderMaterial} = useMaterials();
  
  const geometry = createSilksGeometry();

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
      uniform sampler2D uDisplacementMap;
      uniform vec3 pA1;
      uniform vec3 pA2;
      uniform vec3 pB1;
      uniform vec3 pB2;
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

      const float range = ${range.toFixed(8)};
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
      
      const float cutTime = ${cutTime.toFixed(8)};
      void main() {
        vec2 virtualXZ = vec2(vUv.x * 2.0 - 1.0, vUv.y * 2.0 - 1.0) * range;
        virtualXZ += uWorldPosition.xz;

        vec4 color = texture2D(uDisplacementMap, vUv);

        vec2 a = pA1.xz;
        vec2 b = pA2.xz;
        vec2 c = pB1.xz;
        vec2 d = pB2.xz;
        if (
          (
            isPointInTriangle(virtualXZ, a, b, c) || isPointInTriangle(virtualXZ, b, d, c)
          ) &&
          (uTime - color.w) > cutTime
        ) {
          color.z = (pA1.y + pA2.y + pB1.y + pB2.y) / 4.;
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
    },
    vertexShader: `\
      precision highp float;
      precision highp int;

      uniform float uTime;
      uniform sampler2D uDisplacementMap;
      uniform sampler2D uNoiseTexture;
      attribute vec3 p;
      attribute vec4 q;
      attribute int segment;
      varying vec2 vUv;
      varying vec2 vUv2;
      varying vec3 vNormal;
      varying float vTimeDiff;
      varying float vIsCut;

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
      const float cutSpeed = 1.;
      const float cutTime = ${cutTime.toFixed(8)};
      void main() {
        vec3 pos = position;
        vUv = uv;
        vUv2 = (p.xz + ${(range).toFixed(8)}) / ${(range * 2).toFixed(8)};

        // time diff
        vec4 displacementColor = texture2D(uDisplacementMap, vUv2);
        vTimeDiff = uTime - displacementColor.w;

        // check for cut
        float segmentStartY = float(segment) * segmentHeight;
        float cutY = displacementColor.z;
        float cutSegmentY = floor(cutY / segmentHeight) * segmentHeight;
        bool isCut = (cutY > 0. && cutY < segmentStartY) &&
          (vTimeDiff < cutTime);
        if (isCut) {
          // handle cut
          vec3 centerOfBlade = vec3(0., (cutSegmentY + topSegmentY) * 0.5, 0.);

          pos -= centerOfBlade;
          vec4 q = quat_from_axis_angle(vec3(1., 0., 0.), uTime * 2. * PI * 0.2);
          pos = rotate_vertex_position(pos, q);
          pos += centerOfBlade;

          vec2 directionXZ = -1. + texture2D(uNoiseTexture, vUv2).xz * 2.;
          vec3 direction = vec3(directionXZ.x, 1., directionXZ.y);
          pos += direction * vTimeDiff * cutSpeed;

          vIsCut = 1.;
        } else {
          vIsCut = 0.;
        }

        // instance offset
        {
          pos = rotate_vertex_position(pos, q);
          pos += p;
        }

        if (!isCut) {
          // handle step displacement
          vec3 displacement = texture2D(uDisplacementMap, vUv2).rgb;
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
      varying float vIsCut;

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

      const float cutTime = ${cutTime.toFixed(8)};
      void main() {
        gl_FragColor = vec4(0., 0., 0., 1.);
        vec4 displacementColor = texture2D(uDisplacementMap, vUv2);

        gl_FragColor.rgb = displacementColor.rgb;

        if (vIsCut > 0.) {
          gl_FragColor.a = max(1. - vTimeDiff, 0.);
        } else {
          gl_FragColor.a = 1.;
        }
      }
    `,
    transparent: true,
  });
  const mesh = new THREE.InstancedMesh(geometry, material, numBlades);
  mesh.frustumCulled = false;

  mesh.update = (timestamp, timeDiff) => {
    _renderDisplacementMap();
    _renderMain(timestamp);
    _flipRenderTargets();
  };

  // XXX
  const scene = useScene();
  const cutMesh = _makeCutMesh();
  cutMesh.frustumCulled = false;
  scene.add(cutMesh);

  mesh.hitAttempt = (position, quaternion) => {
    const pointA1 = position.clone()
      .add(new THREE.Vector3(-1, -1.2, -0.1).applyQuaternion(quaternion));
    const pointA2 = position.clone()
      .add(new THREE.Vector3(-0.7, -1.2, -1.5).applyQuaternion(quaternion));
    const pointB1 = position.clone()
      .add(new THREE.Vector3(1, -1.2, -0.1).applyQuaternion(quaternion));
    const pointB2 = position.clone()
      .add(new THREE.Vector3(0.7, -1.2, -1.5).applyQuaternion(quaternion));
    [pointA1, pointA2, pointB1, pointB2].forEach((point, i) => {
      point.toArray(cutMesh.geometry.attributes.position.array, i * 3);
    });
    cutMesh.geometry.attributes.position.needsUpdate = true;

    const timestamp = performance.now();
    _renderCut(pointA1, pointA2, pointB1, pointB2, timestamp);
    _flipRenderTargets();
  };
  return mesh;
};

export default () => {
  const hitManager = useHitManager();

  const mesh = _makeSilksMesh();

  useFrame(({timestamp, timeDiff}) => {
    mesh.update(timestamp, timeDiff);
  });

  hitManager.addEventListener('hitattempt', e => {
    const {type, args} = e.data;
    if (type === 'sword') {
      const {
        position,
        quaternion,
        // hitHalfHeight,
        // hitRadius,
      } = args;
      // console.log('draw cut', e.data, position.toArray().join(','), quaternion.toArray().join(','), cutMesh.geometry);

      mesh.hitAttempt(position, quaternion);
    }
  });
  
  return mesh;
};