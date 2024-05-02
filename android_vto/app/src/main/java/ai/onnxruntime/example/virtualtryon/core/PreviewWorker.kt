package ai.onnxruntime.example.virtualtryon.core

import ai.onnxruntime.example.virtualtryon.core.VTO.preview
import android.content.Context
import android.util.Log
import androidx.work.Worker
import androidx.work.WorkerParameters

class PreviewWorker(appContext: Context, workerParams: WorkerParameters):
    Worker(appContext, workerParams) {

    private var totalModel: TotalModel? = null

    init {
        System.loadLibrary("native_lib")
    }

    override fun doWork(): Result {
        val fileName = inputData.getString("fileName")!!

        // Disassemble the video into frames
        // and save them to the frames directory

        val videoName = fileName.substringBefore(".mp4")

        VideoHandler.disassemble(videoName)

        // Execute preview
        Log.d("PreviewWorker", "Preview Start")

        val rootPath = FileSource.PREVIEW_RESULT_PATH
        totalModel = TotalModel(applicationContext.assets, false)
        preview(rootPath, videoName, totalModel)

        totalModel!!.delete()
        Log.i("End", "Preview End")

        return Result.success()
    }
}