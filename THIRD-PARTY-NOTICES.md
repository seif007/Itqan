# Third-party notices

The Itqān app is distributed with the components below. Each component keeps
its own license; nothing here changes those terms.

## The application itself

The app's own source code is licensed under the **MIT License** — see
[`LICENSE`](LICENSE) at the repository root.

That MIT grant covers the code only. As shipped, the app also runs the NPL-1.1
speech model described below, and NPL-1.1 governs the application *as a whole*:
no sales, paid features, subscriptions, or advertising. The packaged app is
therefore for non-commercial use even though its source code is MIT.

## Speech-recognition model

**FastConformer-Quran — int8 ONNX graph + token map**

- Authors: Muno459 / Quran-Lab
- Upstream: <https://huggingface.co/Muno459/fastconformer-quran> (gated)
- Public mirror used by the app:
  <https://huggingface.co/saif96/itqan-fastconformer-quran-int8>
- License: **Quran-Lab No-Profit License, Version 1.1 (NPL-1.1)**
- Full text: [`app/src/main/assets/stt_model/LICENSE`](app/src/main/assets/stt_model/LICENSE)

NPL-1.1 is a non-commercial, share-alike license. The model is provided free of
charge and may not be sold, put behind a paywall, or used in a feature that is
charged for or carries advertising. Redistributions must keep the same license
and include the license text. The app downloads the weights from the public
mirror on first use; the license file ships inside the APK.

The weights are a fine-tune and int8 ONNX export produced by the upstream
authors. This project redistributes them unchanged.

### Base model

The upstream model is derived from
**NVIDIA `stt_ar_fastconformer_hybrid_large_pcd_v1.0`**
(<https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0>),
licensed under **CC BY 4.0**
(<https://creativecommons.org/licenses/by/4.0/>). The upstream authors
quantized and fine-tuned it; this project made no changes to the weights.

### Training data (not redistributed here)

The upstream model card lists these datasets. They are not included in the app
or in the model mirror.

| Dataset | License |
|---|---|
| [`tarteel-ai/everyayah`](https://huggingface.co/datasets/tarteel-ai/everyayah) | CC BY 4.0 for the dataset; licensing of individual recordings is undocumented |
| [`tarteel-ai/tlog`](https://huggingface.co/datasets/tarteel-ai/tlog) | CC BY 4.0 |
| [`obadx/muaalem-annotated-v3`](https://huggingface.co/datasets/obadx/muaalem-annotated-v3) | MIT |

## ONNX Runtime

**Microsoft ONNX Runtime for Android 1.17.1**

- <https://github.com/microsoft/onnxruntime>
- License: MIT
- Copyright (c) Microsoft Corporation.

## Android and Kotlin libraries

The app uses AndroidX, Jetpack Compose, Kotlin coroutines, OkHttp, Retrofit,
Moshi, and Room. These are distributed under the Apache License 2.0 (AndroidX,
OkHttp, Retrofit, Room), the Apache License 2.0 with the Kotlin runtime
exception (Kotlin), and the Apache License 2.0 (Moshi). Their license texts are
available from the respective upstream projects and are included in the APK's
dependency metadata.
