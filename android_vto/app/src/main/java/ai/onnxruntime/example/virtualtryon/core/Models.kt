package ai.onnxruntime.example.virtualtryon.core

class TotalModel(assetManager: android.content.res.AssetManager, isVto: Boolean) {
    private var modelsAddr: Long

    init {
        System.loadLibrary("native_lib")
        modelsAddr = newModels() //jni call to create c++ reference and returns address

        if (!isVto) {
            parsingModelInit(modelsAddr, assetManager)
        } else {
            vtoModelInit(modelsAddr, assetManager)
        }
    }

    private external fun newModels(): Long
    private external fun parsingModelInit(vtoAddr: Long, mg: android.content.res.AssetManager)
    private external fun vtoModelInit(vtoAddr: Long, mg: android.content.res.AssetManager)

    private external fun deleteModels(vtoAddr: Long)
    private external fun inferenceVTO(
        vtoAddr: Long,
        in_image: ByteArray,
        inCloth: ByteArray,
        inEdge: ByteArray,
        inParsing: ByteArray,
        bb: IntArray,
        newSize: IntArray,
        paddings: IntArray,
        corrSize: IntArray
    ): Array<ByteArray>

    private external fun inferenceParsing(parsingAddr: Long, inbitmap: ByteArray): ByteArray

    /**
     * makes jni call to delete c++ reference
     */
    fun delete() {
        deleteModels(modelsAddr) //jni call to delete c++ reference
        modelsAddr = 0 //set address to 0
    }

    @Throws(Throwable::class)
    protected fun finalize() {
        delete()
    }

    /**
     * return address of c++ reference
     */
    fun getmodelsAddr(): Long {
        return modelsAddr //return address
    }

    /**
     * //makes jni call to proces frames
     */
    fun runVto(
        in_image: ByteArray,
        inCloth: ByteArray,
        inEdge: ByteArray,
        inParsing: ByteArray,
        bb: IntArray,
        newSize: IntArray,
        paddings: IntArray,
        corrSize: IntArray
    ): Array<ByteArray> {
        return inferenceVTO(
            modelsAddr,
            in_image,
            inCloth,
            inEdge,
            inParsing,
            bb,
            newSize,
            paddings,
            corrSize
        ) //jni call to proces frames
    }

    fun runParsing(input: ByteArray): ByteArray {
        return inferenceParsing(modelsAddr, input) //jni call to process frames
    }
}