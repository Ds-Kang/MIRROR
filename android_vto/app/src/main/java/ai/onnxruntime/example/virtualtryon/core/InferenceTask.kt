package ai.onnxruntime.example.virtualtryon.core

data class InferenceTask(
    val inferenceType: String,
    val fileName: String?,
    val clothImageUrl: String?
)
