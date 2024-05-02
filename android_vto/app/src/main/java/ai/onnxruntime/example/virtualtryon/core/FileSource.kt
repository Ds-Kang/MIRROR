package ai.onnxruntime.example.virtualtryon.core

import android.util.Log
import java.io.File

/**
 * Singleton class to hold the file io source
 * Below is the directory structure
 *
 * mirror
 *   - /original
 *      - /videos
 *         - /video1.mp4
 *      - /frames
 *         - /1
 *            - /1.jpg
 *      - /preview_results
 *         - /video1
 *            - /1.jpg
 *      - /clothes
 *         - /1.jpg
 *      - /edges
 *         - /1.jpg
 *   - /runtime_results
 *      - /video1.mp4
 */
object FileSource {
    private var ROOT_DIR = "data/local/tmp/mirror"

    val ORIGINAL_PATH: String
        get() = "$ROOT_DIR/original"
    val ORIGINAL_VIDEO_PATH: String
        get() = "$ORIGINAL_PATH/videos"
    val ORIGINAL_FRAME_PATH: String
        get() = "$ORIGINAL_PATH/frames"
    val PREVIEW_RESULT_PATH: String
        get() = "$ORIGINAL_PATH/preview_results"
    val CLOTHES_PATH: String
        get() = "$ORIGINAL_PATH/clothes"
    val EDGES_PATH: String
        get() = "$ORIGINAL_PATH/edges"
    val LIT_INFO_PATH: String
        get() = "$PREVIEW_RESULT_PATH/lit_info"
    val DNN_INFO_PATH: String
        get() = "$PREVIEW_RESULT_PATH/dnn_info"
    val RUNTIME_RESULTS_PATH: String
        get() = "$ROOT_DIR/runtime_results"

    fun getOriginalFramesPath(videoName: String): String {
        return "$ORIGINAL_FRAME_PATH/$videoName"
    }

    fun getPreviewResultsPath(videoName: String): String {
        return "$PREVIEW_RESULT_PATH/$videoName"
    }

    fun getDefaultVideoName(): String {
        val dir = File(ORIGINAL_VIDEO_PATH, "default")
        if (!dir.exists()) {
            Log.e("FileSource", "Default video does not exist")
            return ""
        }
        val files = dir.listFiles()
        if (files == null || files.isEmpty()) {
            Log.e("FileSource", "Default video does not exist")
            return ""
        }
        return files[0].name.substringBefore(".mp4")
    }

    fun initializeRootDir(rootDir: String) {
        ROOT_DIR = rootDir
        for (path in arrayOf(
            ORIGINAL_VIDEO_PATH,
            ORIGINAL_FRAME_PATH,
            PREVIEW_RESULT_PATH,
            CLOTHES_PATH,
            EDGES_PATH,
            LIT_INFO_PATH,
            DNN_INFO_PATH,
            RUNTIME_RESULTS_PATH
        )) {
            val dir = File(path)
            if (!dir.exists()) {
                dir.mkdirs()
            }
        }
    }
}