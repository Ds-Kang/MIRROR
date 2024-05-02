package ai.onnxruntime.example.virtualtryon.camerascreen

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaScannerConnection
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.LifecycleOwner
import ai.onnxruntime.example.virtualtryon.R
import ai.onnxruntime.example.virtualtryon.core.FileSource
import ai.onnxruntime.example.virtualtryon.core.InferenceService
import ai.onnxruntime.example.virtualtryon.core.InferenceWorker
import ai.onnxruntime.example.virtualtryon.core.PreviewWorker
import android.content.Intent
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.concurrent.Executors

class CameraFragment : Fragment() {

    private lateinit var previewView: PreviewView
    private lateinit var preview: Preview
    private lateinit var safeContext: Context
    private lateinit var imageCapture: ImageCapture
    private lateinit var recorder: Recorder
    private lateinit var videoCapture: VideoCapture<Recorder>

    val cameraExecutor = Executors.newSingleThreadExecutor()

    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var recording: Recording? = null

    override fun onAttach(context: Context) {
        super.onAttach(context)
        safeContext = context
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_camera, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        previewView = view.findViewById(R.id.previewView)

        // Request camera permissions
        val cameraPermissionCheck =
            ContextCompat.checkSelfPermission(safeContext, Manifest.permission.CAMERA)
        if (cameraPermissionCheck == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(Manifest.permission.CAMERA),
                1000
            )
        }

        startCamera()

        val recordButton = view.findViewById<ImageView>(R.id.recordButton)
        recordButton.setOnClickListener {
            onRecordVideo(recordButton)
        }

        val switchCameraButton = view.findViewById<ImageView>(R.id.switchCameraButton)
        switchCameraButton.setOnClickListener {
            onSwitchCamera()
        }
    }

    // Initialize camera
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(safeContext)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider)
        }, ContextCompat.getMainExecutor(safeContext))
    }

    // Bind camera to preview
    private fun bindPreview(cameraProvider: ProcessCameraProvider) {
        // Select rear camera as default
        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

        // Build required use cases
        preview = Preview.Builder().build()
        recorder =
            Recorder.Builder().setExecutor(cameraExecutor)
                .build()
        videoCapture = VideoCapture.withOutput(recorder)

        try {
            // Get preview output surface
            preview.setSurfaceProvider(previewView.surfaceProvider)
            // Unbind the use-cases before rebinding them
            cameraProvider.unbindAll()
            // Bind camera to lifecycle
            val camera = cameraProvider.bindToLifecycle(
                this as LifecycleOwner,
                cameraSelector,
                preview,
                videoCapture
            )
        } catch (exc: Exception) {
            Log.e("CameraFragment", "Use case binding failed", exc)
        }
    }

    // Switch camera between front and back
    private fun onSwitchCamera() {
        lensFacing = when (lensFacing) {
            CameraSelector.LENS_FACING_FRONT -> CameraSelector.LENS_FACING_BACK
            CameraSelector.LENS_FACING_BACK -> CameraSelector.LENS_FACING_FRONT
            else -> CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }

    // Take picture
    private fun onTakePicture() {
        // Add content values
        val contentValues = ContentValues()
        contentValues.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        contentValues.put(
            MediaStore.Images.Media.DISPLAY_NAME,
            SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        )
        contentValues.put(MediaStore.Images.Media.DATE_TAKEN, System.currentTimeMillis())
        contentValues.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)

        // Set output file options
        val outputFileOptions = ImageCapture.OutputFileOptions.Builder(
            safeContext.contentResolver,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        ).build()

        // Take picture with preset thread
        imageCapture.takePicture(
            outputFileOptions,
            cameraExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    val outputURI = outputFileResults.savedUri
                    MediaScannerConnection.scanFile(
                        safeContext,
                        arrayOf(outputURI?.path), null
                    ) { _, _ ->
                        Toast.makeText(safeContext, "Photo saved", Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Toast.makeText(safeContext, "Error saving photo", Toast.LENGTH_SHORT).show()
                }
            })
    }

    // Record video
    private fun onRecordVideo(view: ImageView) {

        // Stop recording if it is already recording
        if (recording != null) {
            view.setImageResource(R.drawable.vector_record_button)
            recording?.stop()
            recording = null
            return
        }

        val fileName = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date()) + ".mp4"

        // Save video to mediaStore (NOT USED)
        val contentValues = ContentValues()
        contentValues.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
        contentValues.put(MediaStore.Video.Media.DISPLAY_NAME, fileName)
        contentValues.put(MediaStore.Video.Media.DATE_TAKEN, System.currentTimeMillis())
        contentValues.put(
            MediaStore.Video.Media.RELATIVE_PATH,
            Environment.DIRECTORY_MOVIES + "/MIRROR"
        )
        val mediaStoreOutputOptions = MediaStoreOutputOptions.Builder(
            safeContext.contentResolver,
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        )
            .setContentValues(contentValues)
            .build()

        // Save video to internal storage
        val defaultFileDir = File(FileSource.ORIGINAL_VIDEO_PATH, "default")
        val source = defaultFileDir.listFiles()?.size?.let {
            if (it == 0) {
                // If there is no default video, make the first video as default
                File(FileSource.ORIGINAL_VIDEO_PATH, "default")
            }
            else {
                // If there is a default video, make the new video as default
                File(FileSource.ORIGINAL_VIDEO_PATH)
            }
        }
        val file = File(source, fileName)
        val internalOutputOptions = FileOutputOptions.Builder(file).build()

        // Configure recorder and start recording to mediaStoreOutput
        recording = videoCapture.output
            .prepareRecording(safeContext, internalOutputOptions)
            .start(ContextCompat.getMainExecutor(safeContext)) { recordEvent ->
                when (recordEvent) {
                    is VideoRecordEvent.Start -> {
                        Toast.makeText(safeContext, "Recording started", Toast.LENGTH_SHORT).show()
                    }

                    is VideoRecordEvent.Finalize -> {
                        Toast.makeText(safeContext, "Video captured", Toast.LENGTH_SHORT).show()
                        Log.i("CameraFragment", "Video saved to ${file.absolutePath}")
                        runPreviewService(fileName)
                    }
                }
            }

        // Change the button icon
        view.setImageResource(
            when (recording) {
                null -> R.drawable.vector_record_button
                else -> R.drawable.vector_stop_record_button
            }
        )
    }

    private fun runPreviewWorker(fileName: String) {
        val data = Data.Builder()
            .putString("type", "preview")
            .putString("fileName", fileName)
            .build()
        val previewWorker = OneTimeWorkRequestBuilder<InferenceWorker>()
            .setInputData(data)
            .build()
        WorkManager.getInstance(safeContext).beginUniqueWork(
            "InferenceWorker",
            ExistingWorkPolicy.REPLACE,
            previewWorker
        ).enqueue()
    }

    private fun runPreviewService(fileName: String) {
        val intent = Intent(safeContext, InferenceService::class.java)
        intent.putExtra("type", "preview")
        intent.putExtra("fileName", fileName)
        safeContext.startForegroundService(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}