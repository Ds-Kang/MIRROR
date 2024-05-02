package ai.onnxruntime.example.virtualtryon.core

import ai.onnxruntime.example.virtualtryon.R
import android.app.NotificationManager
import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.Worker
import androidx.work.WorkerParameters

class InferenceWorker(appContext: Context, workerParams: WorkerParameters):
    Worker(appContext, workerParams) {

    private var totalModel: TotalModel? = null
    private val appContext = appContext

    override fun doWork(): Result {
        val inferenceType = inputData.getString("type")!!

        if (inferenceType == "preview") {

            val fileName = inputData.getString("fileName")!!

            // Disassemble the video into frames
            // and save them to the frames directory
            val videoName = fileName.substringBefore(".mp4")
            VideoHandler.disassemble(videoName)

            // Execute preview
            Log.d("PreviewWorker", "Preview Start")
            val rootPath = FileSource.PREVIEW_RESULT_PATH
            totalModel = TotalModel(applicationContext.assets, false)
            VTO.preview(rootPath, videoName, totalModel)

            Log.i("End", "Preview End")
            return Result.success()

        } else if (inferenceType == "runtime") {

            val startTime = System.currentTimeMillis()

            val clothImageUrl = inputData.getString("clothImageUrl")!!

            // Execute runtime
            Log.d("RuntimeWorker", "Runtime Start")
            val rootPath = FileSource.RUNTIME_RESULTS_PATH
            val videoName = FileSource.getDefaultVideoName()
            val bitmap = BitmapFactory.decodeStream(appContext.contentResolver.openInputStream(Uri.parse(clothImageUrl)))
            totalModel = TotalModel(applicationContext.assets, true)
            VTO.runtime(rootPath, videoName, bitmap, totalModel)

            val endTime = System.currentTimeMillis()

            Log.i("RuntimeWorker", "Runtime Inference End in ${(endTime - startTime) / 1000} s")
            sendNotification()
            return Result.success()

        } else {
            return Result.failure()
        }
    }

    override fun onStopped() {
        super.onStopped()
        totalModel?.delete()
    }

    private fun sendNotification() {
        /**
         * Sends a notification to the user when the work is completed successfully or when it failed.
         */
        val notificationManager = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        val notificationBuilder = NotificationCompat.Builder(applicationContext, "ort_virtual_try_on")
            .setContentTitle("Try-On Complete")
            .setContentText("Your video is ready!")
            .setSmallIcon(R.drawable.ic_launcher_foreground)

        // Notify the user
        notificationManager.notify(1001, notificationBuilder.build())
    }

}