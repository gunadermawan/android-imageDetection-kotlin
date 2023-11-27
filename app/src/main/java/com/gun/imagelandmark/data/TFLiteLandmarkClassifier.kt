package com.gun.imagelandmark.data

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.gun.imagelandmark.domain.Classification
import com.gun.imagelandmark.domain.LandmarkClassifier
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier

class TFLiteLandmarkClassifier(
    private val context: Context,
    private val threshold: Float = 0.5f,
    private val maxResult: Int = 1,
) : LandmarkClassifier {
    private var classifier: ImageClassifier? = null
    private fun setUpClassifier() {
        val baseOptions = BaseOptions.builder()
            .setNumThreads(2)
            .build()
        val options = ImageClassifier.ImageClassifierOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(maxResult)
            .setScoreThreshold(threshold)
            .build()
        try {
            classifier =
                ImageClassifier.createFromFileAndOptions(context, "tensofflow-asia.tflite", options)

        } catch (e: IllegalStateException) {
            e.printStackTrace()
        }
    }

    override fun classify(bitmap: Bitmap, rotation: Int): List<Classification> {
        if (classifier == null) {
            setUpClassifier()
        }
        val imageProcessor = org.tensorflow.lite.support.image.ImageProcessor.Builder().build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(rotation)).build()
        val result = classifier?.classify(tensorImage, imageProcessingOptions)
        return result?.flatMap { classification ->
            classification.categories.map {
                Classification(
                    name = it.displayName,
                    score = it.score
                )
            }
        }?.distinctBy { it.name } ?: emptyList()
    }

    private fun getOrientationFromRotation(rotation: Int): ImageProcessingOptions.Orientation {
        return when (rotation) {
            android.view.Surface.ROTATION_270 -> ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            android.view.Surface.ROTATION_90 -> ImageProcessingOptions.Orientation.TOP_LEFT
            android.view.Surface.ROTATION_180 -> ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            else -> ImageProcessingOptions.Orientation.RIGHT_TOP
        }
    }
}