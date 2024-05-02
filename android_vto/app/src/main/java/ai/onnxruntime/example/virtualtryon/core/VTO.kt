package ai.onnxruntime.example.virtualtryon.core

import android.content.ContentResolver
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.provider.MediaStore
import android.provider.MediaStore.Audio.Media
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import java.io.File
import java.io.Serializable
import java.nio.ByteBuffer
import java.text.DecimalFormat

object VTO {
    private val scope = CoroutineScope(Job() + Dispatchers.Main)
    private var limited: Boolean = false
    private var totalModel: TotalModel? = null
    private var tmpInfo: TmpInfo = TmpInfo()
    private var dnnInfo: DNNInfo = DNNInfo()
    private var litInfo: LITInfo = LITInfo()
    private val parsingSize: Int = 7
    private var nccSub: Float = 10.0F
    private var tpsThreshold: Float = 0.1F

    init {
        System.loadLibrary("native_lib")
    }

    external fun getBoundingBox(parseJava: ByteArray, parsingSize: Int): IntArray
    external fun cropParse(parseJava: ByteArray, bbJava: IntArray): ByteArray
    external fun cropTorsoParse(parseJava: ByteArray, bbJava: IntArray): ByteArray
    external fun byteToInt(byteJava: ByteArray): IntArray
    external fun genToFrameMask(
        genJava: ByteArray,
        input2: IntArray,
        paddingsJava: IntArray,
        corrSizeJava: IntArray,
        frameJava: ByteArray,
        maskJava: ByteArray
    ): ByteArray

    external fun genWithMapMask(
        genJava: ByteArray,
        input2: IntArray,
        paddingsJava: IntArray,
        corrSizeJava: IntArray,
        gridJava: FloatArray,
        frameJava: ByteArray,
        maskJava: ByteArray
    ): ByteArray

    external fun calcMap(
        bbJava: IntArray,
        paddingsJava: IntArray,
        corrSizeJava: IntArray,
        gridJava: FloatArray
    ): FloatArray

    external fun orbExtraction(
        input1: IntArray,
        input2: ByteArray,
        parseJava: ByteArray
    ): FloatArray

    external fun opticalFlow(
        input1: FloatArray,
        input2: FloatArray,
        input3: ByteArray,
        input4: ByteArray
    ): FloatArray

    external fun gridGen(
        input1: FloatArray,
        input2: FloatArray,
        input3: IntArray,
        input4: IntArray
    ): Array<FloatArray>

    external fun getOFrame(input1: ByteArray, input4: ByteArray, bbJava: IntArray): ByteArray
    external fun calcNCC(
        oFrameJava: ByteArray,
        bbJava: IntArray,
        mapXYJava: FloatArray,
        frameJava: ByteArray,
        parsingJava: ByteArray
    ): Float

    external fun convertToNV21(rgbFrameJava: ByteArray): ByteArray

    external fun getMaskFromRGB(inputImageJava: ByteArray, height: Int, width: Int): ByteArray

    internal fun preview(rootPath: String, videoNumber: String, totalModel: TotalModel?) {
        val frameCount = File(FileSource.ORIGINAL_FRAME_PATH, videoNumber).listFiles()?.size!!
        Log.i("Frame Count", "Preview with total frames count: $frameCount")

        val df = DecimalFormat("0000")
        var dnnIndex = BooleanArray(frameCount)

        var dnnCount = 0
        var litCount = 0
        var lightweightTime = 0L
        var dnnTime = 0L

        litInfo.nccSub = nccSub
        litInfo.tpsThreshold = tpsThreshold

        totalModel?.let {
            this.totalModel = it
        }

        for (i in 1 until frameCount + 1) {
            val frameNum = df.format(i).toString()
//            Log.i("Frame", "Frame: $frameNum conversion start!")
            val startTime = SystemClock.uptimeMillis()

            var refBitmap = readVideoFrame(videoNumber, frameNum)
            var refArray = preProcess(refBitmap, 3, 960, 540)
            var isDNN = false

            if (dnnInfo.firstP0.isNotEmpty()) {
                // Inference using Lightweight Image Transformation
                previewLightweight(refArray, videoNumber, frameNum)
                litCount += 1
            }
            if (i == 1 || litInfo.ncc < litInfo.nccThreshold || litInfo.tpsError > litInfo.tpsThreshold) {
                dnnCount += 1
                previewDNN(refArray, videoNumber, frameNum)
                isDNN = true
                dnnIndex[i - 1] = isDNN
            }
            Log.i(
                "Preview Frame",
                "Frame: $frameNum\tInference Time: ${(SystemClock.uptimeMillis() - startTime)}ms\tType: ${if (isDNN) "DNN" else "Lightweight"}"
            )
            if (isDNN) {
                dnnTime += (SystemClock.uptimeMillis() - startTime)
            } else {
                lightweightTime += (SystemClock.uptimeMillis() - startTime)
            }
//            Log.i("DNN", "Skip DNN: ${(!isDNN)}")
//            Log.i("Inference Time", "Inference Time: ${(SystemClock.uptimeMillis() - startTime)}ms")
//            Log.i("Counts", "DNN: $dnnCount, Lightweight: $litCount")
        }
        saveIndex(videoNumber, dnnIndex)
        Log.i(
            "Preview", "Preview Inference End\n" +
                    "DNN: $dnnCount, Lightweight: $litCount\n" +
                    "Average Lightweight Time: ${lightweightTime / (frameCount - dnnCount)}ms\n" +
                    "Average DNN Time: ${dnnTime / dnnCount}ms\n" +
                    "Total Time: ${lightweightTime + dnnTime}ms"
        )
    }


    internal fun runtime(
        rootPath: String,
        videoNumber: String,
        clothImage: Bitmap,
        totalModel: TotalModel?
    ) {
        Log.i("Runtime", "Runtime Inference Start")

        val videoEncoder = VideoEncoder(object : VideoEncoder.IVideoEncoderCallback {
            override fun onEncodingComplete(outputFile: File?) {
            }
        })

        if (!File("$rootPath/$videoNumber").exists()) {
            File("$rootPath/$videoNumber").mkdirs()
        }

        // If the original video has been processed, make output_1.mp4, output_2.mp4, ...
        var outputNumber = 1
        while (File("$rootPath/$videoNumber/output_$outputNumber.mp4").exists()) {
            outputNumber += 1
        }
        videoEncoder.startEncoding(
            540,
            960,
            File("$rootPath/$videoNumber/output_$outputNumber.mp4")
        )

        val df = DecimalFormat("0000")
        val result = Result()
        val dnnIndex = readIndex(videoNumber) ?: return

        // Read Cloth, Edge
        val clothBitmap = removeTransparency(clothImage)
        val clothArray: ByteArray = preProcessCloth(clothBitmap, 3, 256, 192)
        val edgeArray: ByteArray = getMaskFromRGB(clothArray, 256, 192)

        var dnnCount = 0
        var litCount = 0
        val frameCount = File(FileSource.ORIGINAL_FRAME_PATH, videoNumber).listFiles()?.size!!
        var lightweightTime = 0L
        var dnnTime = 0L

        totalModel?.let {
            this.totalModel = it
        }

        for (i in 1 until frameCount + 1) {
            val frameNum = df.format(i).toString()
//            Log.i("Frame", "Frame: $frameNum conversion start!")
            val startTime = SystemClock.uptimeMillis()

            var refBitmap = readVideoFrame(videoNumber, frameNum)
            var refArray = preProcess(refBitmap, 3, 960, 540)

            if (!dnnIndex[i - 1]) {
                // Inference using Lightweight Image Transformation
                runtimeLightweight(refArray, result, videoNumber, frameNum)
                litCount += 1
            }
            if (dnnIndex[i - 1]) {
                dnnCount += 1
                runtimeDNN(refArray, clothArray, edgeArray, result, videoNumber, frameNum)
            }
            Log.i(
                "Runtime Frame",
                "Frame: $frameNum\tInference Time: ${(SystemClock.uptimeMillis() - startTime)}ms\tType: ${if (dnnIndex[i - 1]) "DNN" else "Lightweight"}"
            )
            if (dnnIndex[i - 1]) {
                dnnTime += (SystemClock.uptimeMillis() - startTime)
            } else {
                lightweightTime += (SystemClock.uptimeMillis() - startTime)
            }
//            Log.i("DNN", "Skip DNN: ${(!dnnIndex[i - 1])}")
//            Log.i("Inference Time", "Inference Time: ${(SystemClock.uptimeMillis() - startTime)}")
//            Log.i("Counts", "DNN: $dnnCount, Lightweight: $litCount")
            val yuvFrame = convertToNV21(result.genImageByte)
            videoEncoder.queueFrame(yuvFrame)
        }
        saveIndex(videoNumber, dnnIndex)
        videoEncoder.stopEncoding()
        Log.i(
            "Runtime", "Runtime Inference End\n" +
                    "DNN: $dnnCount, Lightweight: $litCount\n" +
                    "Average Lightweight Time: ${lightweightTime / (frameCount - dnnCount)}ms\n" +
                    "Average DNN Time: ${dnnTime / dnnCount}ms\n" +
                    "Total Time: ${lightweightTime + dnnTime}ms"
        )
    }

    internal fun previewDNN(refArray: ByteArray, videoNumber: String, frameNumber: String) {
        val parseArray = totalModel!!.runParsing(refArray)
        dnnInfo.firstBB = getBoundingBox(parseArray, parsingSize)
        dnnInfo.croppedParsing = cropParse(parseArray, dnnInfo.firstBB)
        dnnInfo.croppedTorsoParsing = cropTorsoParse(parseArray, dnnInfo.firstBB)
        dnnInfo.firstP0 = orbExtraction(dnnInfo.firstBB, refArray, parseArray)
        dnnInfo.ratio = getRatio(dnnInfo.firstBB, 256.0F, 192.0F)
        dnnInfo.newSize = getResizeSize(dnnInfo.firstBB, dnnInfo.ratio)
        dnnInfo.paddings = getPaddings(dnnInfo.newSize, 256, 192)
        dnnInfo.corrSize = getCorrSize(dnnInfo.firstBB, dnnInfo.ratio)

        writeDnnInfo(dnnInfo, videoNumber, frameNumber)

        litInfo.bb = dnnInfo.firstBB

        litInfo.ncc = 1.0F
        litInfo.tpsError = 0.0F

        tmpInfo.oldFrame = refArray
        tmpInfo.p0 = dnnInfo.firstP0

        tmpInfo.oFrame = getOFrame(refArray, dnnInfo.croppedTorsoParsing, dnnInfo.firstBB)
    }

    internal fun runtimeDNN(
        refArray: ByteArray,
        clothArray: ByteArray,
        edgeArray: ByteArray,
        result: Result,
        videoNumber: String,
        frameNumber: String
    ) {
        val tmpDnnInfo = readDnnInfo(videoNumber, frameNumber)
        if (tmpDnnInfo != null) {
            dnnInfo = tmpDnnInfo.copy()
        }

        litInfo.bb = dnnInfo.firstBB

        val modelOut: Array<ByteArray> = totalModel!!.runVto(
            refArray,
            clothArray,
            edgeArray,
            dnnInfo.croppedParsing,
            dnnInfo.firstBB,
            dnnInfo.newSize,
            dnnInfo.paddings,
            dnnInfo.corrSize
        )
        tmpInfo.gen = modelOut[0]
        tmpInfo.mask = modelOut[1]

        result.genImageByte = genToFrameMask(
            tmpInfo.gen,
            dnnInfo.firstBB,
            dnnInfo.paddings,
            dnnInfo.corrSize,
            refArray,
            tmpInfo.mask
        )
        tmpInfo.oFrame = getOFrame(refArray, tmpInfo.mask, dnnInfo.firstBB)
    }


    internal fun previewLightweight(refArray: ByteArray, videoNumber: String, frameNumber: String) {
        var mapXY: FloatArray

        val p1_with_first = opticalFlow(tmpInfo.p0, dnnInfo.firstP0, tmpInfo.oldFrame, refArray)

        tmpInfo.p0 = getP1(p1_with_first, p1_with_first.size / 2)
        // good points
        dnnInfo.firstP0 = getFirstP0(p1_with_first, p1_with_first.size / 2)

        val diff = getDiff(tmpInfo.p0, dnnInfo.firstP0)
        litInfo.bb = moveBB(dnnInfo.firstBB, diff)

        val grid_output = gridGen(tmpInfo.p0, dnnInfo.firstP0, diff, litInfo.bb)
        litInfo.grid = grid_output[0]
        mapXY = calcMap(litInfo.bb, dnnInfo.paddings, dnnInfo.corrSize, litInfo.grid)
        litInfo.tpsError = grid_output[1][0]

        litInfo.ncc =
            calcNCC(tmpInfo.oFrame, litInfo.bb, mapXY, refArray, dnnInfo.croppedTorsoParsing)
        if (litInfo.nccThreshold == -1.0F) {
            val moved_p0 = moveP0(dnnInfo.firstP0, litInfo.nccSub)
            val tmpDiff = getDiff(dnnInfo.firstP0, dnnInfo.firstP0)
            val grid_output_tmp = gridGen(moved_p0, dnnInfo.firstP0, tmpDiff, dnnInfo.firstBB)
            val mapXY_tmp =
                calcMap(dnnInfo.firstBB, dnnInfo.paddings, dnnInfo.corrSize, grid_output_tmp[0])

            litInfo.nccThreshold =
                calcNCC(
                    tmpInfo.oFrame,
                    dnnInfo.firstBB,
                    mapXY_tmp,
                    refArray,
                    dnnInfo.croppedTorsoParsing
                )
        }
        if (litInfo.ncc > litInfo.nccThreshold && litInfo.tpsError < litInfo.tpsThreshold) {
            writeLitInfo(litInfo, videoNumber, frameNumber)
        }
        Log.i(
            "Threshold",
            "NCC Threshold: ${litInfo.nccThreshold}, TPS Threshold: ${litInfo.tpsThreshold}"
        )
        Log.i("Errors", "NCC Error: ${litInfo.ncc}, TPS Error: ${litInfo.tpsError}")
        tmpInfo.oldFrame = refArray
    }


    internal fun runtimeLightweight(
        refArray: ByteArray,
        result: Result,
        videoNumber: String,
        frameNumber: String
    ) {
        var mapXY: FloatArray

        val tmpLitInfo = readLitInfo(videoNumber, frameNumber, limited)
        if (tmpLitInfo != null) {
            litInfo = tmpLitInfo.copy()
        }
        mapXY = calcMap(litInfo.bb, dnnInfo.paddings, dnnInfo.corrSize, litInfo.grid)

        tmpInfo.oldFrame = refArray
        result.genImageByte = genWithMapMask(
            tmpInfo.gen,
            litInfo.bb,
            dnnInfo.paddings,
            dnnInfo.corrSize,
            mapXY,
            refArray,
            tmpInfo.mask
        )
    }

    internal data class Result(
        var genImageByte: ByteArray = ByteArray(0),
        var processTimeMs: Long = 0
    ) : Serializable


    internal data class DNNInfo(
        var firstBB: IntArray = IntArray(0),
        var croppedParsing: ByteArray = ByteArray(0),
        var croppedTorsoParsing: ByteArray = ByteArray(0),
        var firstP0: FloatArray = FloatArray(0),
        var ratio: Float = 0.0F,
        var newSize: IntArray = IntArray(0),
        var paddings: IntArray = IntArray(0),
        var corrSize: IntArray = IntArray(0)
    ) : Serializable


    internal data class LITInfo(
        var grid: FloatArray = FloatArray(0),
        var bb: IntArray = IntArray(0),
        var ncc: Float = 1.0F,
        var nccThreshold: Float = -1.0F,
        var nccSub: Float = -1.0F,
        var tpsError: Float = 0.0F,
        var tpsThreshold: Float = 100.0F
    ) : Serializable


    internal data class TmpInfo(
        var p0: FloatArray = FloatArray(0),
        var gen: ByteArray = ByteArray(0),
        var mask: ByteArray = ByteArray(0),
        var oldFrame: ByteArray = ByteArray(0),
        var oFrame: ByteArray = ByteArray(0)
    ) : Serializable
}