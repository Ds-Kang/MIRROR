package ai.onnxruntime.example.virtualtryon.core

import ai.onnxruntime.example.virtualtryon.MainActivity
import ai.onnxruntime.example.virtualtryon.R
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.IBinder
import android.util.Log
import android.widget.Toast
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Queue
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.ThreadFactory

class InferenceService : Service() {

    private var totalModel: TotalModel? = null
    private val taskQueue: Queue<InferenceTask> = ConcurrentLinkedQueue()
    private var isProcessing = false

    override fun onBind(intent: Intent?): IBinder? {
        TODO("Not yet implemented")
    }

    override fun onCreate() {
        // Set up IO directories
        FileSource.initializeRootDir(applicationContext.filesDir.absolutePath)

        // Create notification channel
        val channelId = "ort_virtual_try_on"
        val notificationManager = getSystemService(android.app.NotificationManager::class.java)
        val notificationChannel = android.app.NotificationChannel(
            channelId,
            "MIRROR",
            NotificationManager.IMPORTANCE_DEFAULT
        )
        notificationManager.createNotificationChannel(notificationChannel)
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("MIRROR")
            .setContentText("Inference in progress")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .build()

        startForeground(1, notification)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val inferenceType = intent?.getStringExtra("type")!!
        val fileName = intent.getStringExtra("fileName")
        val clothImageUrl = intent.getStringExtra("clothImageUrl")
        taskQueue.add(InferenceTask(inferenceType, fileName, clothImageUrl))
        Log.i("InferenceService", "Inference type: $inferenceType")

        if (!isProcessing) {
            isProcessing = true
            processNextTask()
        }

        return START_NOT_STICKY
    }

    private fun processNextTask() {
        if (taskQueue.isEmpty()) {
            stopSelf()
            return
        }

        val task = taskQueue.peek()

        val maxPriorityThread = ThreadFactory {
            Thread(it).apply {
                priority = Thread.MAX_PRIORITY
            }
        }

        val maxPriorityDispatcher =
            Executors.newSingleThreadExecutor(maxPriorityThread).asCoroutineDispatcher()

        CoroutineScope(maxPriorityDispatcher).launch {
            if (task != null) {
                when (task.inferenceType) {
                    "preview" -> {
                        runPreview(task.fileName!!)
                    }
                    "runtime" -> {
                        runRuntime(task.clothImageUrl!!)
                    }
                }
            }
            // Send notification when inference is complete
            sendNotification()
            onTaskCompleted()
        }
    }

    private fun onTaskCompleted() {
        taskQueue.poll()
        isProcessing = false
        processNextTask()
    }


    private fun runPreview(fileName: String) {
        totalModel = TotalModel(applicationContext.assets, false)
        val videoName = fileName.substringBefore(".mp4")
        VideoHandler.disassemble(videoName)
        val rootPath = FileSource.PREVIEW_RESULT_PATH
        VTO.preview(rootPath, videoName, totalModel)
        stopSelf()
    }

    private fun runRuntime(clothImageUrl: String) {
        totalModel = TotalModel(applicationContext.assets, true)
        val rootPath = FileSource.RUNTIME_RESULTS_PATH
        val videoName = FileSource.getDefaultVideoName()

        val bitmap: Bitmap
        bitmap = getBitmapFromUrl(clothImageUrl)
        VTO.runtime(rootPath, videoName, bitmap, totalModel)

        stopSelf()
    }

    override fun onDestroy() {
        // Release ML model
        totalModel?.delete()
    }

    private fun getBitmapFromUrl(url: String): Bitmap {
        Log.i("InferenceService", "Getting bitmap from url: $url")
        return BitmapFactory.decodeStream(
            applicationContext.contentResolver.openInputStream(
                Uri.parse(
                    url
                )
            )
        )
    }

    private fun sendNotification() {
        /**
         * Sends a notification to the user when the work is completed successfully or when it failed.
         */
        val intent = Intent(applicationContext, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        } // Opens MainActivity when notification is clicked

        val pendingIntent = PendingIntent.getActivity(
            applicationContext,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notificationManager =
            applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        val notificationBuilder =
            NotificationCompat.Builder(applicationContext, "ort_virtual_try_on")
                .setContentTitle("Try-On Complete")
                .setContentText("Your video is ready!")
                .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                .setSmallIcon(R.drawable.ic_launcher_foreground)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)

        // Notify the user
        notificationManager.notify(1001, notificationBuilder.build())
    }
}