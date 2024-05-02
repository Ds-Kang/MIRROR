package ai.onnxruntime.example.virtualtryon.core

import android.content.Context
import android.util.Log
import androidx.work.Data
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.arthenica.mobileffmpeg.FFmpeg
import com.arthenica.mobileffmpeg.Config
import java.io.File

object VideoHandler {

    private const val VIDEO_WIDTH = 540
    private const val VIDEO_HEIGHT = 960

    fun disassemble(videoName: String) {
        /**
         * Disassembles video to image frames of size 960 x 540
         * and saves them to the directory of identical name.
         * Image file names are sequenced as:
         * 0001.png, 0002.png, 0003.png, ... */

        val videoFile = File(FileSource.ORIGINAL_VIDEO_PATH, "$videoName.mp4")
        val videoPath = videoFile.absolutePath

        val framesPath = FileSource.getOriginalFramesPath(videoName)
        val framesDir = File(framesPath)
        if (!framesDir.exists()) {
            framesDir.mkdirs()
        }

        Log.i(Config.TAG, "FFmpeg process started with arguments:\nInput: $videoPath \nOutput: $framesDir")
        val command = "-i $videoPath -vf \"scale=$VIDEO_WIDTH:$VIDEO_HEIGHT,fps=30\" $framesDir/%04d.png"

        when (val rc = FFmpeg.execute(command)) {
            Config.RETURN_CODE_SUCCESS -> {
                Log.i(Config.TAG, "Command execution completed successfully.")
            }
            Config.RETURN_CODE_CANCEL -> {
                Log.i(Config.TAG, "Command execution cancelled by user.")
            }
            else -> {
                Log.e(Config.TAG, String.format("Command execution failed with rc=%d and the output below.", rc))
                Config.printLastCommandOutput(Log.INFO)
            }
        }
        Log.i(Config.TAG, "FFmpeg process ended.")
    }

    fun assemble(videoName: String) {
        /**
         * Assembles frames into video of frame rate 30
         * and saves them to the directory of identical name
         * Video file is named as: videoName.mp4
         * This method is for development */

        val framesPath = FileSource.getOriginalFramesPath(videoName)
        val videoPath = "${FileSource.ORIGINAL_VIDEO_PATH}/$videoName.mp4"

        Log.i(Config.TAG, "FFmpeg process started with arguments:\nInput: $framesPath \nOutput: $videoPath")
        val command = "-framerate 30 -i $framesPath/%04d.png -vf \"scale=$VIDEO_WIDTH:$VIDEO_HEIGHT\" -c:v mpeg4 -pix_fmt yuv420p $videoPath"

        when (val rc = FFmpeg.execute(command)) {
            Config.RETURN_CODE_SUCCESS -> {
                Log.i(Config.TAG, "Command execution completed successfully.")
            }
            Config.RETURN_CODE_CANCEL -> {
                Log.e(Config.TAG, "Command execution cancelled by user.")
            }
            else -> {
                Log.e(Config.TAG, String.format("Command execution failed with rc=%d and the output below.", rc))
                Config.printLastCommandOutput(Log.INFO)
            }
        }
        Log.i(Config.TAG, "FFmpeg process ended.")
    }

    fun preview(context: Context, videoName: String) {
        /**
         * Runs manual preview inference
         * This method is for development
         */

        val data = Data.Builder()
            .putString("fileName", videoName)
            .build()
        val previewWorker = OneTimeWorkRequestBuilder<PreviewWorker>()
            .setInputData(data)
            .build()
        WorkManager.getInstance(context).beginUniqueWork(
            "preview",
            androidx.work.ExistingWorkPolicy.REPLACE,
            previewWorker
        ).enqueue()
    }

    fun runtime(context: Context, clothImageUrl: String) {
        /**
         * Runs runtime inference
         * This method is for development
         */

        val data = Data.Builder()
            .putString("clothImageUrl", clothImageUrl)
            .build()
        val runtimeWorker = OneTimeWorkRequestBuilder<RuntimeWorker>()
            .setInputData(data)
            .build()
        WorkManager.getInstance(context).enqueue(runtimeWorker)
    }
}