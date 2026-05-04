package com.babelsoftware.airnote.data.repository

import androidx.annotation.StringRes
import com.babelsoftware.airnote.R

data class LlmModelInfo(
    val name: String,
    @StringRes val displayNameResId: Int
)

object LlmModels {
    val supportedModels = listOf(
        LlmModelInfo(
            name = "llama3-70b-8192",
            displayNameResId = R.string.llm_models_llama3_70b
        ),
        LlmModelInfo(
            name = "llama3-8b-8192",
            displayNameResId = R.string.llm_models_llama3_8b
        ),
        LlmModelInfo(
            name = "mixtral-8x7b-32768",
            displayNameResId = R.string.llm_models_mixtral_8x7b
        ),
        LlmModelInfo(
            name = "gemma2-9b-it",
            displayNameResId = R.string.llm_models_gemma2_9b
        )
    )

    val defaultEndpoint = "https://api.groq.com/openai/v1"
}
