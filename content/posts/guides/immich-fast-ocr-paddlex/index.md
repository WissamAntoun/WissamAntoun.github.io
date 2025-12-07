---
title: "Fast Inference for Immich ML OCR with PaddleX"
date: 2025-12-07T17:00:00+08:00
hero: images/immich23-1024x576.png
description: "A guide to deploying PaddleX for fast OCR inference on GPU for Immich ML."
theme: Toha
menu:
    sidebar:
        name: "Fast Inference for Immich ML OCR with PaddleX"
        identifier: immich-fast-ocr-paddlex
        parent: guides
        weight: 105
tags:
- Immich
- OCR
- Inference
- GPU
- PaddleX
- Guide
- Self-Hosting
- Bug
---

This guide provides step-by-step instructions to set up PaddleX for fast OCR inference on GPU for Immich ML. By following these steps, you can significantly improve the performance of OCR tasks in your Immich setup.

{{< alert type="success" >}}
~1s inference time per image with 10 concurrent requests on RTX 3080ti 12GB GPU with PaddleX vs 80s+ with Onnxruntime GPU execution provider.
{{< /alert >}}

{{< alert type="success" >}}
All code snippets and configuration files mentioned in this guide can be found in this [GitHub repository](https://github.com/WissamAntoun/immich-ml-fast-ocr-paddlex).
{{< /alert >}}

## Context

{{< alert type="warning" >}}
**Note:** Test with Immich Version: 2.3.1
{{< /alert >}}

[Immich](https://immich.app/) ML's [inference engine](https://github.com/immich-app/immich/tree/main/machine-learning) uses [Onnxruntime](https://onnxruntime.ai/) as the default backend for model inference. However, for OCR tasks which were introduced in Immich 2.3.x, users have reported very slow GPU performance sometimes even slower than CPU inference (see [Reddit discussion](https://www.reddit.com/r/immich/comments/1omj5ue/ocr_job_is_slow/), [GitHub Issue #23462](https://github.com/immich-app/immich/issues/23462)) even when using a powerful GPU or the mobilde version of the OCR model.

This performance issue was caused by the Onnxruntime GPU execution engine not being optimized for dynamic input sizes, which is common in OCR tasks. This issue is mentioned in the [RapidOCR Docs](https://rapidai.github.io/RapidOCRDocs/main/blog/2022/09/24/onnxruntime-gpu%E6%8E%A8%E7%90%86/#onnxruntime-gpu) and was brought up to the Immich team in [GitHub Issue #23462](https://github.com/immich-app/immich/issues/23462#issuecomment-3567416353) and this GitHub issue [comment](https://github.com/immich-app/immich/issues/23462#issuecomment-3583530978) by me.

I'm also facing issue with my remote machine with RTX 3080ti 12GB running server model with 10 concurrent requests. The GPU is at a 100% with 170W/350W power consumption. 10 images are taking around 2 minutes to finish, even with concurrency of 5 or 1 it's slow.

Since I also have 100K+ photos in my Immich instance, I wanted to find a solution to speed up the OCR inference process or this would take forever to finish and costed an arm and a leg in electricity bills.

## Solution: Routing OCR Inference to PaddleX OCR Pipeline

After investigating various options, I found that [PaddleX](https://paddlepaddle.github.io/PaddleX/latest/) provides a highly optimized OCR pipeline for the same model used in Immich ML, which can leverage GPU acceleration effectively even with dynamic input sizes. Cool!

My approach requires no building from source, it only relies on mounting modified files into the Immich ML Docker and PaddleX containers to route the OCR inference to a PaddleX service.

### Step 1: Set Up PaddleX OCR Service

First, we need to set up a PaddleX OCR service that will handle the OCR inference requests. To do this, we will create a Docker container running PaddleX with the OCR model.

1. **Create a new directory for the PaddleX Docker setup or clone this project's [GitHub repository](https://github.com/WissamAntoun/immich-ml-fast-ocr-paddlex).**
2. **Create a `Dockerfile` with the following content found here [PaddleX Dockerfile](https://github.com/WissamAntoun/immich-ml-fast-ocr-paddlex/blob/main/paddle_ocr/Dockerfile):**

```Dockerfile
# paddle_ocr/Dockerfile
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.4-paddlepaddle3.2.0-gpu-cuda11.8-cudnn8.9-trt8.6

RUN python -m pip install "paddleocr[all]==3.3.2"

RUN paddlex --install serving
```

Note that i tested with PaddleX 3.3.4 and PaddleOCR 3.3.2 versions and made my changes based on these versions. Any future versions might require adjustments.
Also since the containers are based on Baidu's registry i wasn't able to check which is the latest version available there and just grabbed the latest one i could find mentioned on github.

3. **Put the following two files in the root of the main directory:**

Paddlex replacements files [`paddlex_ocr_pipeline.py`](https://github.com/WissamAntoun/immich-ml-fast-ocr-paddlex/blob/main/paddle_ocr/paddlex_ocr_pipeline.py) and [`paddlex_ocr_result.py`](https://github.com/WissamAntoun/immich-ml-fast-ocr-paddlex/blob/main/paddle_ocr/paddlex_ocr_result.py).

**Key changes from original pipeline.py:**

- Line 362: Extract `dt_scores` from detection results
- Line 372: Pass `dt_scores` to the OCR processing data
- Line 383, 388: Add `dt_scores` to the zip iteration to ensure scores are properly propagated

```diff
--- pipeline.py
+++ pipeline.py
@@ -359,6 +359,7 @@
             )

             dt_polys_list = [item["dt_polys"] for item in det_results]
+            dt_scores_list = [item["dt_scores"] for item in det_results]

             dt_polys_list = [self._sort_boxes(item) for item in dt_polys_list]

@@ -368,6 +369,7 @@
                     "page_index": page_index,
                     "doc_preprocessor_res": doc_preprocessor_res,
                     "dt_polys": dt_polys,
+                    "dt_scores": dt_scores,
                     "model_settings": model_settings,
                     "text_det_params": text_det_params,
                     "text_type": self.text_type,
@@ -378,11 +380,12 @@
                     "rec_polys": [],
                     "vis_fonts": [],
                 }
-                for input_path, page_index, doc_preprocessor_res, dt_polys in zip(
+                for input_path, page_index, doc_preprocessor_res, dt_polys, dt_scores in zip(
                     batch_data.input_paths,
                     batch_data.page_indexes,
                     doc_preprocessor_results,
                     dt_polys_list,
+                    dt_scores_list,
                 )
             ]
```

**Key changes from original result.py:**

- Line 210: Add `dt_scores` to the data dictionary to ensure scores are included in the results

```diff
--- result.py
+++ result.py
@@ -210,6 +210,7 @@
         if self["model_settings"]["use_doc_preprocessor"]:
             data["doc_preprocessor_res"] = self["doc_preprocessor_res"].json["res"]
         data["dt_polys"] = self["dt_polys"]
+        data["dt_scores"] = self["dt_scores"]
         data["text_det_params"] = self["text_det_params"]
         data["text_type"] = self["text_type"]
         if "textline_orientation_angles" in self:
```

4. **Create an `OCR.yaml` configuration file for the PaddleX OCR pipeline with the following content found here [OCR.yaml](https://github.com/WissamAntoun/immich-ml-fast-ocr-paddlex/blob/main/paddle_ocr/OCR.yaml).**

Here I used the `PP-OCRv5_server_det` and `PP-OCRv5_server_rec` models as they fit in my 12GB GPU memory while providing good accuracy and speed. You can experiment with other models as needed depending on your GPU or language requirements.

The default values for the parameters are taken from the PaddleX and from Immich ML repo. Note that some parameters can be overriden from the Immich side when calling the ML service and will take precedence over these values.

```yaml
pipeline_name: OCR

text_type: general

use_doc_preprocessor: True
use_textline_orientation: True

SubPipelines:
  DocPreprocessor:
    pipeline_name: doc_preprocessor
    use_doc_orientation_classify: True
    use_doc_unwarping: True
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null
      DocUnwarping:
        module_name: image_unwarping
        model_name: UVDoc
        model_dir: null

# followed defaults from here https://github.com/PaddlePaddle/PaddleX/blob/release/3.3/paddlex/inference/models/text_detection/predictor.py#L127
# and here https://github.com/immich-app/immich/blob/main/machine-learning/immich_ml/models/ocr/detection.py#L34
SubModules:
  TextDetection:
    module_name: text_detection
    model_name: PP-OCRv5_server_det
    model_dir: null
    limit_side_len: 960
    limit_type: max
    max_side_limit: 4000
    thresh: 0.3
    box_thresh: 0.5
    unclip_ratio: 1.6
  TextLineOrientation:
    module_name: textline_orientation
    model_name: PP-LCNet_x1_0_textline_ori
    model_dir: null
    batch_size: 6
  TextRecognition:
    module_name: text_recognition
    model_name: PP-OCRv5_server_rec
    model_dir: null
    batch_size: 6
    score_thresh: 0.0
    return_word_box: True
```

After setting up these files, you should have the following structure:

```
paddle_ocr/
├── Dockerfile
├── paddlex_ocr_pipeline.py
├── paddlex_ocr_result.py
└── OCR.yaml
```

### Step 2: Build and Run the PaddleX Docker Container

Now, build and run the PaddleX Docker container with the modified files mounted. Check the `paddle-ocr` service in the provided [docker-compose.yml](https://github.com/WissamAntoun/immich-ml-fast-ocr-paddlex/blob/main/docker-compose.yml) for reference.

```yaml
  paddle-ocr:
    container_name: paddle_ocr
    build:
      context: ./paddle_ocr
      dockerfile: Dockerfile
    volumes:
      - ./model-cache/:/root/.paddlex/official_models/
      # - ./paddle_ocr/paddlex_ocr_pipeline.py:/usr/local/lib/python3.10/dist-packages/paddlex/inference/pipelines/ocr/pipeline.py
      # - ./paddle_ocr/paddlex_ocr_result.py:/usr/local/lib/python3.10/dist-packages/paddlex/inference/pipelines/ocr/result.py
      - ./paddle_ocr/paddlex_ocr_pipeline.py:/root/PaddleX/paddlex/inference/pipelines/ocr/pipeline.py
      - ./paddle_ocr/paddlex_ocr_result.py:/root/PaddleX/paddlex/inference/pipelines/ocr/result.py
      - ./paddle_ocr/OCR.yaml:/OCR.yaml
    ports:
      - "8866:8080"
    command: "paddlex --serve --pipeline /OCR.yaml"
    restart: unless-stopped
    healthcheck:
      disable: false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities:
                - gpu
```

Note that if you change the paddlex image version in the Dockerfile, and paddlex gets reinstalled and upgraded, the path to the `pipeline.py` and `result.py` files might change to the newly installed version. You will need to adjust the volume mounts accordingly. I included two possible paths in the example above, uncomment the correct one based on your installation.

Use the following commands to build and run the container:

```bash
mkdir -p model-cache
docker-compose build paddle-ocr
docker-compose up -d paddle-ocr
```

To verify that the PaddleX OCR service is running correctly, you use the following command to download and send a test image for OCR:

Note: The first time you run this command, it might take a while as the model weights and some fonts will be downloaded and cached.

```bash
curl -sL "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/images/Banner.png" -o ./ocr_test.png

python -c "import requests, base64; r = requests.post('http://localhost:8866/ocr/', json={'file': base64.b64encode(open('./ocr_test.png', 'rb').read()).decode('utf-8'), 'fileType': 1, 'visualize': False}); result = r.json(); texts = result['result']['ocrResults'][0]['prunedResult']['rec_texts'] if r.status_code == 200 and 'result' in result else []; print(f'✓ OCR server is working - Found {len(texts)} text items:\n' + '\n'.join([f\"  - {text}\" for text in texts]) if texts else f'✗ OCR server failed: {r.status_code}')"
```

You should see output similar to:

```
✓ OCR server is working - Found 16 text items:
  - PP-ChatOCRv4
  - PP-StructureV3
  - 文档解析
  - PP-StructureV3
  - 户飞架
  - PaddleOCR
  - PP-OCRv5
  - PaddleOCR3.0
  - <I>010101010101
  - PP-OCRv5
  - 文字识别
  - oduct launched
  - TextRecognition&DocParsingToolkit
  - もじ文字にんしき認識です
  - Qi
  - √s
```

### Step 3: Modify Immich ML to Use PaddleX OCR Service

Finally, we need to modify the Immich ML service to route OCR inference requests to the PaddleX OCR service instead of using Onnxruntime.

1. **Similar to the PaddleX setup, we will mount modified files into the Immich ML Docker container to override the OCR pipeline and result handling.**

The file to be modified in the `immich_ml/main.py` file. You can find the modified version here [main.py](https://github.com/WissamAntoun/immich-ml-fast-ocr-paddlex/blob/main/immich_ml_main.py).

**Key changes from original main.py:**

- Added `OCR_PADDLEX_URL` environment variable to specify the PaddleX OCR service URL.
- Modified the `preload_models` function to skip loading OCR models if `OCR_PADDLEX_URL` is set.
- Updated the `/inference/` endpoint to route OCR requests to the PaddleX service when `OCR_PADDLEX_URL` is set.
- Implemented `run_paddlex_inference` function to handle OCR requests using the PaddleX service.
- Added `process_paddlex_results` function to process the results returned by the PaddleX OCR service and format them to match Immich ML's expected output structure.
- Updated error handling to provide detailed logs and HTTP exceptions for PaddleX OCR inference failures.

```diff
@@ -1,5 +1,6 @@
 import asyncio
 import gc
+import io
 import os
 import signal
 import threading
@@ -10,7 +11,10 @@
 from typing import Any, AsyncGenerator, Callable, Iterator
 from zipfile import BadZipFile

+import cv2
 import orjson
+import numpy as np
+from numpy.typing import NDArray
 from fastapi import Depends, FastAPI, File, Form, HTTPException
 from fastapi.responses import ORJSONResponse, PlainTextResponse
 from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf, NoSuchFile
@@ -43,6 +47,11 @@
 lock = threading.Lock()
 active_requests = 0
 last_called: float | None = None
+OCR_PADDLEX_URL = os.environ.get("OCR_PADDLEX_URL", None)
+if OCR_PADDLEX_URL is not None:
+    log.info(
+        f"OCR_PADDLEX_URL is set to: {OCR_PADDLEX_URL}. Will use PaddleX OCR if set instead of RapidOCR."
+    )


 @asynccontextmanager
@@ -113,19 +122,21 @@
             ModelTask.FACIAL_RECOGNITION,
         )

-    if preload.ocr.detection is not None:
+    if preload.ocr.detection is not None and OCR_PADDLEX_URL is None:
         await load_models(
             preload.ocr.detection,
             ModelType.DETECTION,
             ModelTask.OCR,
         )

-    if preload.ocr.recognition is not None:
+    if preload.ocr.recognition is not None and OCR_PADDLEX_URL is None:
         await load_models(
             preload.ocr.recognition,
             ModelType.RECOGNITION,
             ModelTask.OCR,
         )
+    if OCR_PADDLEX_URL is not None:
+        log.info(f"Using PaddleX OCR pipeline for OCR requests from {OCR_PADDLEX_URL}")

     if preload.clip_fallback is not None:
         log.warning(
@@ -198,10 +209,158 @@
         inputs = text
     else:
         raise HTTPException(400, "Either image or text must be provided")
-    response = await run_inference(inputs, entries)
+    if OCR_PADDLEX_URL is not None and (entries[0][0]["task"] == ModelTask.OCR):
+        log.warning("Using PaddleX OCR pipeline for OCR requests.")
+        response = await run_paddlex_inference(inputs, entries)
+    else:
+        response = await run_inference(inputs, entries)
+        log.warning(f"Inference Output: {response}")
     return ORJSONResponse(response)


+def process_paddlex_results(
+    paddlex_response_dict: dict[str, Any],
+    min_recognition_score: float,
+    image_width: int,
+    image_height: int,
+) -> dict[str, Any]:
+    texts = paddlex_response_dict["rec_texts"]
+    boxes = np.array(paddlex_response_dict["dt_polys"], dtype=np.float32)
+    if boxes.shape[0] == 0:
+        return {
+            "ocr": {
+                "box": np.empty(0, dtype=np.float32),
+                "text": [],
+                "boxScore": np.empty(0, dtype=np.float32),
+                "textScore": np.empty(0, dtype=np.float32),
+            },
+            "imageWidth": image_width,
+            "imageHeight": image_height,
+        }
+
+    boxes[:, :, 0] /= image_width
+    boxes[:, :, 1] /= image_height
+    box_scores = np.array(paddlex_response_dict["dt_scores"], dtype=np.float32)
+    text_scores = np.array(paddlex_response_dict["rec_scores"])
+    valid_text_score_idx = text_scores > min_recognition_score
+    # remove the ids where text is empty string
+    for i, text in enumerate(texts):
+        if text.strip() == "":
+            valid_text_score_idx[i] = False
+    valid_score_idx_list = valid_text_score_idx.tolist()
+
+    boxes = boxes.reshape(-1, 8)[valid_text_score_idx].reshape(-1)
+    texts = [texts[i] for i in range(len(texts)) if valid_score_idx_list[i]]
+    boxScore = box_scores[valid_text_score_idx]
+    textScore = text_scores[valid_text_score_idx]
+
+    return {
+        "ocr": {
+            "box": boxes,
+            "text": texts,
+            "boxScore": boxScore,
+            "textScore": textScore,
+        },
+        "imageWidth": image_width,
+        "imageHeight": image_height,
+    }
+
+
+async def run_paddlex_inference(
+    payload: Image | str, entries: InferenceEntries
+) -> InferenceResponse:
+    import base64
+    import requests
+    import traceback
+
+    start_time = time.time()
+    response: InferenceResponse = {}
+
+    try:
+        # Although PaddleX supports overriding some parameters per request,
+        # I only translated the ones used in Immich ML for consistency.
+        # The rest of the parameters can be changed in the ./paddle_ocr/OCR.yaml config file.
+        # Be Careful that whatever is here will override the config file settings.
+
+        if isinstance(payload, Image):
+            image_width = payload.width
+            image_height = payload.height
+        else:
+            image_width = 0
+            image_height = 0
+
+        buffered = io.BytesIO()
+        payload.save(buffered, format="JPEG")
+        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
+
+        req_payload = {
+            "file": encoded_image,
+            "fileType": 1,
+            "returnWordBox": False,
+            "visualize": False,  #  default is False
+        }
+
+        max_resolution = entries[0][0]["options"].get("maxResolution", None)
+        if max_resolution is not None:
+            req_payload["textDetLimitSideLen"] = max_resolution
+
+        min_detection_score = entries[0][0]["options"].get("minScore", None)
+        if min_detection_score is not None:
+            req_payload["textDetBoxThresh"] = min_detection_score
+
+        min_recognition_score = entries[1][0]["options"].get("minScore", None)
+
+        # Send the request asynchronously
+        paddlex_response = await run(
+            lambda: requests.post(OCR_PADDLEX_URL, json=req_payload)
+        )
+        log.info(f"Response Status Code: {paddlex_response.status_code}")
+        # log.info(f"Raw PaddleX OCR response: {paddlex_response.text}")
+
+        if paddlex_response.status_code != 200:
+            log.error(
+                f"PaddleX OCR inference failed with status code {paddlex_response.status_code} and message: {paddlex_response.text}"
+            )
+            raise HTTPException(
+                500,
+                f"PaddleX OCR inference failed with status code {paddlex_response.status_code} and message: {paddlex_response.text}",
+            )
+
+        paddlex_response_dict = paddlex_response.json()["result"]["ocrResults"][0][
+            "prunedResult"
+        ]
+
+        response = await run(
+            process_paddlex_results,
+            paddlex_response_dict,
+            min_recognition_score,
+            image_width,
+            image_height,
+        )
+        end_time = time.time()
+        log.info(f"PaddleX OCR inference took {end_time - start_time:.3f} seconds.")
+        # log.warning(f"OCR PaddleX Output: {response}")
+        return response
+    except HTTPException:
+        # Re-raise HTTPExceptions as-is
+        raise
+    except requests.exceptions.RequestException as e:
+        error_msg = f"PaddleX OCR request failed: {str(e)}"
+        stack_trace = traceback.format_exc()
+        log.error(f"{error_msg}\n{stack_trace}")
+        raise HTTPException(500, f"{error_msg}\nStack trace:\n{stack_trace}")
+    except (KeyError, IndexError, ValueError) as e:
+        error_msg = f"Failed to parse PaddleX OCR response: {str(e)}"
+        stack_trace = traceback.format_exc()
+        log.error(f"{error_msg}\n{stack_trace}")
+        raise HTTPException(500, f"{error_msg}\nStack trace:\n{stack_trace}")
+    except Exception as e:
+        error_msg = f"Unexpected error during PaddleX OCR inference: {str(e)}"
+        stack_trace = traceback.format_exc()
+        log.error(f"{error_msg}\n{stack_trace}")
+        raise HTTPException(500, f"{error_msg}\nStack trace:\n{stack_trace}")
+
+
 async def run_inference(
     payload: Image | str, entries: InferenceEntries
 ) -> InferenceResponse:
```

2. **Update your Immich ML Docker Compose configuration to mount the modified `main.py` file.**

Check the `immich_ml` service in the provided [docker-compose.yml](https://github.com/WissamAntoun/immich-ml-fast-ocr-paddlex/blob/main/docker-compose.yml)

```yaml
    immich-machine-learning:
    container_name: immich_machine_learning
    # For hardware acceleration, add one of -[armnn, cuda, rocm, openvino, rknn] to the image tag.
    # Example tag: ${IMMICH_VERSION:-release}-cuda
    image: ghcr.io/immich-app/immich-machine-learning:${IMMICH_VERSION:-release}-cuda
    volumes:
      - ./model-cache:/cache
      - ./immich_ml_main.py:/usr/src/immich_ml/main.py
    ports:
      - "33003:3003" # modify to your desired port mapping
    environment:
      - OCR_PADDLEX_URL=http://paddle_ocr:8080/ocr # Pointing to the PaddleX OCR service
    restart: always
    healthcheck:
      disable: false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities:
                - gpu
```

Start or restart the Immich ML container to apply the changes:

```bash
docker-compose up -d immich-machine-learning
```

And that's it! You have successfully set up PaddleX for fast OCR inference on GPU for Immich ML. You should now see a significant improvement in OCR inference times when processing images in your Immich instance.
