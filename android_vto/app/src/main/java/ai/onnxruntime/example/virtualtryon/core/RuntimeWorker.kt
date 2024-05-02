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
import java.io.File

class RuntimeWorker(appContext: Context, workerParams: WorkerParameters):
    Worker(appContext, workerParams)  {

    private var totalModel: TotalModel? = null
    private val appContext = appContext

    init {
        System.loadLibrary("native_lib")
    }

    override fun doWork(): Result {
        val clothImageUrl = inputData.getString("clothImageUrl")!!
        Log.d("RuntimeWorker", "Runtime Inference Start with $clothImageUrl")

        val rootPath = FileSource.RUNTIME_RESULTS_PATH
        val videoName = FileSource.getDefaultVideoName()
        val bitmap = BitmapFactory.decodeStream(appContext.contentResolver.openInputStream(Uri.parse(clothImageUrl)))
        totalModel = TotalModel(applicationContext.assets, true)
        VTO.runtime(rootPath, videoName, bitmap, totalModel)

        Log.i("RuntimeWorker", "Runtime Inference End")
        sendNotification()
        return Result.success()
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