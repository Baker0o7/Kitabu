/*
 * Copyright (c) 2025 Babel Software.
 *
 *
 */

package com.babelsoftware.airnote.data.repository

import android.util.Log
import com.babelsoftware.airnote.R
import com.babelsoftware.airnote.data.provider.StringProvider
import com.babelsoftware.airnote.domain.model.ChatMessage
import com.babelsoftware.airnote.domain.model.Note
import com.babelsoftware.airnote.domain.model.Participant
import com.babelsoftware.airnote.domain.repository.SettingsRepository
import com.google.gson.annotations.SerializedName
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.TranslateRemoteModel
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslatorOptions
import io.ktor.client.HttpClient
import io.ktor.client.plugins.ClientRequestException
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsChannel
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.ktor.utils.io.readUTF8Line
import jakarta.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

enum class AiMode {
    NOTE_ASSISTANT,
    CREATIVE_MIND,
    ACADEMIC_RESEARCHER,
    PROFESSIONAL_STRATEGIST
}

enum class AiAction {
    IMPROVE_WRITING,
    SUMMARIZE,
    MAKE_SHORTER,
    MAKE_LONGER,
    CHANGE_TONE,
    TRANSLATE
}

enum class AiTone {
    FORMAL,
    BALANCED,
    FRIENDLY
}

enum class AiAssistantAction {
    GIVE_IDEA,
    CONTINUE_WRITING,
    CHANGE_PERSPECTIVE,
    PROS_AND_CONS,
    CREATE_TODO_LIST,
    SIMPLIFY,
    SUGGEST_A_TITLE
}

data class AiActionPlan(
    @SerializedName("thought")
    val thought: String?,

    @SerializedName("actions")
    val actions: List<AiActionCommand>,

    @SerializedName("response_message")
    val response_message: String
)

data class AiActionCommand(
    @SerializedName("action_type")
    val action_type: String,
    @SerializedName("title")
    val title: String?,
    @SerializedName("content")
    val content: String?,
    @SerializedName("tasks")
    val tasks: List<String>?,
    @SerializedName("name")
    val name: String?,
    @SerializedName("iconName")
    val iconName: String?,
    @SerializedName("note_title")
    val note_title: String?,
    @SerializedName("folder_name")
    val folder_name: String?,
    @SerializedName("response")
    val response: String?,
    @SerializedName("new_content")
    val new_content: String?,
    @SerializedName("search_term")
    val search_term: String?
)

// --- OpenAI-compatible API data classes ---

@Serializable
private data class ChatMessageDto(
    @SerialName("role") val role: String,
    @SerialName("content") val content: String
)

@Serializable
private data class ChatCompletionRequest(
    @SerialName("model") val model: String,
    @SerialName("stream") val stream: Boolean,
    @SerialName("messages") val messages: List<ChatMessageDto>,
    @SerialName("temperature") val temperature: Double? = null,
    @SerialName("max_tokens") val maxTokens: Int? = null
)

@Serializable
private data class ChatCompletionDelta(
    @SerialName("content") val content: String? = null
)

@Serializable
private data class ChatCompletionChoice(
    @SerialName("delta") val delta: ChatCompletionDelta? = null,
    @SerialName("message") val message: ChatMessageDto? = null,
    @SerialName("index") val index: Int = 0,
    @SerialName("finish_reason") val finishReason: String? = null
)

@Serializable
private data class ChatCompletionResponse(
    @SerialName("choices") val choices: List<ChatCompletionChoice>
)

@Serializable
private data class ChatCompletionChunk(
    @SerialName("choices") val choices: List<ChatCompletionChoice>
)

@Serializable
private data class ModelInfo(
    @SerialName("id") val id: String
)

@Serializable
private data class ModelListResponse(
    @SerialName("data") val data: List<ModelInfo>
)

class GeminiRepository @Inject constructor(
    private val settingsRepository: SettingsRepository,
    private val stringProvider: StringProvider,
    private val httpClient: HttpClient,
    private val json: Json
) {
    private companion object {
        const val DEFAULT_TEMPERATURE = 0.7
        const val CREATIVE_TEMPERATURE = 0.9
    }

    class ApiKeyMissingException(message: String) : Exception(message)

    private suspend fun getApiEndpoint(): String {
        val settings = settingsRepository.settings.first()
        return settings.llmApiEndpoint.ifBlank { LlmModels.defaultEndpoint }
    }

    private suspend fun getChatCompletionsUrl(): String {
        val endpoint = getApiEndpoint().trimEnd('/')
        return "$endpoint/chat/completions"
    }

    suspend fun validateApiKey(apiKey: String, modelName: String): Result<Unit> = withContext(Dispatchers.IO) {
        if (apiKey.isBlank()) {
            return@withContext Result.failure(ApiKeyMissingException("API key cannot be empty."))
        }
        return@withContext try {
            val url = getChatCompletionsUrl()
            val request = ChatCompletionRequest(
                model = modelName,
                stream = false,
                messages = listOf(ChatMessageDto(role = "user", content = "Hello")),
                maxTokens = 5
            )
            val response = httpClient.post(url) {
                header("Authorization", "Bearer $apiKey")
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            if (response.status == HttpStatusCode.OK) {
                Result.success(Unit)
            } else {
                Result.failure(Exception("API returned status: ${response.status.value}"))
            }
        } catch (e: ClientRequestException) {
            if (e.response.status == HttpStatusCode.Unauthorized) {
                Result.failure(Exception("API key is invalid or unauthorized."))
            } else {
                Result.failure(e)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Result.failure(e)
        }
    }

    suspend fun validatePerplexityApiKey(apiKey: String): Result<Unit> = withContext(Dispatchers.IO) {
        if (apiKey.isBlank()) {
            return@withContext Result.failure(ApiKeyMissingException("The Perplexity API key cannot be empty."))
        }

        val settings = settingsRepository.settings.first()
        val modelToValidate = settings.selectedPerplexityModelName

        try {
            val request = ChatCompletionRequest(
                model = modelToValidate,
                stream = false,
                messages = listOf(ChatMessageDto(role = "user", content = "Hello")),
                maxTokens = 5
            )

            val response = httpClient.post("https://api.perplexity.ai/chat/completions") {
                header("Authorization", "Bearer $apiKey")
                contentType(ContentType.Application.Json)
                setBody(request)
            }

            return@withContext if (response.status == HttpStatusCode.OK) {
                Result.success(Unit)
            } else {
                Result.failure(Exception("Perplexity API returned status: ${response.status.value}"))
            }
        } catch (e: ClientRequestException) {
            if (e.response.status == HttpStatusCode.Unauthorized) {
                Result.failure(Exception("Perplexity API key is invalid or unauthorized."))
            } else {
                Result.failure(e)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Result.failure(e)
        }
    }


    suspend fun processAiAction(text: String, action: AiAction, tone: AiTone? = null, apiKey: String, aiMode: AiMode = AiMode.NOTE_ASSISTANT): Result<String> {
        if (apiKey.isBlank()) {
            return Result.failure(ApiKeyMissingException(stringProvider.getString(R.string.error_no_user_api_key)))
        }
        if (action == AiAction.CHANGE_TONE) {
            require(tone != null) { "CHANGE_TONE action requires a tone." }
        }

        val modelName = settingsRepository.settings.first().selectedModelName
        val temperature = if (aiMode == AiMode.CREATIVE_MIND) CREATIVE_TEMPERATURE else DEFAULT_TEMPERATURE

        val systemPrompt = when (aiMode) {
            AiMode.NOTE_ASSISTANT -> stringProvider.getString(R.string.system_prompt_note_assistant)
            AiMode.CREATIVE_MIND -> stringProvider.getString(R.string.system_prompt_creative_mind)
            AiMode.ACADEMIC_RESEARCHER -> stringProvider.getString(R.string.system_prompt_academic_researcher)
            AiMode.PROFESSIONAL_STRATEGIST -> stringProvider.getString(R.string.system_prompt_professional_strategist)
        }

        val userPrompt = when (action) {
            AiAction.IMPROVE_WRITING -> stringProvider.getString(R.string.prompt_improve_writing, text)
            AiAction.SUMMARIZE -> stringProvider.getString(R.string.prompt_summarize, text)
            AiAction.MAKE_SHORTER -> stringProvider.getString(R.string.prompt_make_shorter, text)
            AiAction.MAKE_LONGER -> stringProvider.getString(R.string.prompt_make_longer, text)
            AiAction.CHANGE_TONE -> {
                val tonePromptResId = when (tone) {
                    AiTone.FORMAL -> R.string.prompt_instruction_tone_formal
                    AiTone.BALANCED -> R.string.prompt_instruction_tone_balanced
                    AiTone.FRIENDLY -> R.string.prompt_instruction_tone_friendly
                    null -> throw IllegalArgumentException("Tone cannot be null for CHANGE_TONE action")
                }
                stringProvider.getString(R.string.prompt_change_tone_template, stringProvider.getString(tonePromptResId), text)
            }
            AiAction.TRANSLATE -> ""
        }

        val url = getChatCompletionsUrl()
        val request = ChatCompletionRequest(
            model = modelName,
            stream = false,
            messages = listOf(
                ChatMessageDto(role = "system", content = systemPrompt),
                ChatMessageDto(role = "user", content = userPrompt)
            ),
            temperature = temperature
        )

        return try {
            val response = httpClient.post(url) {
                header("Authorization", "Bearer $apiKey")
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            val body = response.bodyAsChannel().readUTF8Line() ?: ""
            val completion = json.decodeFromString<ChatCompletionResponse>(body)
            val text = completion.choices.firstOrNull()?.message?.content
            if (text != null) {
                Result.success(text)
            } else {
                Result.failure(Exception("Empty response from API."))
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Result.failure(e)
        }
    }

    fun generatePerplexityChatResponse(
        history: List<ChatMessage>,
        apiKey: String
    ): Flow<String> = flow {
        if (apiKey.isBlank()) {
            throw ApiKeyMissingException("Perplexity API key not set.")
        }

        val settings = settingsRepository.settings.first()
        val modelToUse = settings.selectedPerplexityModelName

        val messages = mutableListOf<ChatMessageDto>()
        messages.add(ChatMessageDto(role = "system", content = "Be helpful and concise."))
        history.filter { !it.isLoading && it.participant != Participant.ERROR }.forEach { msg ->
            messages.add(ChatMessageDto(
                role = if (msg.participant == Participant.USER) "user" else "assistant",
                content = msg.text
            ))
        }

        val requestBody = ChatCompletionRequest(
            model = modelToUse,
            stream = true,
            messages = messages
        )

        try {
            val response = httpClient.post("https://api.perplexity.ai/chat/completions") {
                header("Authorization", "Bearer $apiKey")
                contentType(ContentType.Application.Json)
                setBody(requestBody)
            }

            val channel = response.bodyAsChannel()

            while (!channel.isClosedForRead) {
                val line = channel.readUTF8Line()
                if (line.isNullOrBlank()) continue

                if (line.startsWith("data:")) {
                    val jsonData = line.substring(5).trim()
                    if (jsonData == "[DONE]") {
                        break
                    }

                    try {
                        val streamResponse = json.decodeFromString<ChatCompletionChunk>(jsonData)
                        val choice = streamResponse.choices.firstOrNull()
                        if (choice != null) {
                            val content = choice.delta?.content
                            if (content != null) {
                                emit(content)
                            }
                        }
                    } catch (e: Exception) {
                        Log.w("PerplexityParseError", "JSON parse error: $jsonData", e)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("PerplexityRequestError", "Perplexity request failed", e)
            throw e
        }

    }.catch {
        Log.e("PerplexityFlowError", "Flow error", it)
        emit(stringProvider.getString(R.string.error_api_request_failed, it.message ?: "Unknown error"))
    }.flowOn(Dispatchers.IO)


    fun processAssistantAction(
        noteName: String,
        noteDescription: String,
        action: AiAssistantAction,
        apiKey: String,
        aiMode: AiMode = AiMode.CREATIVE_MIND
    ): Flow<String> = flow {
        if (apiKey.isBlank()) {
            throw ApiKeyMissingException(stringProvider.getString(R.string.error_no_user_api_key))
        }

        val modelName = settingsRepository.settings.first().selectedModelName
        val temperature = if (aiMode == AiMode.CREATIVE_MIND) CREATIVE_TEMPERATURE else DEFAULT_TEMPERATURE

        val systemPrompt = when (aiMode) {
            AiMode.NOTE_ASSISTANT -> stringProvider.getString(R.string.system_prompt_note_assistant)
            AiMode.CREATIVE_MIND -> stringProvider.getString(R.string.system_prompt_creative_mind)
            AiMode.ACADEMIC_RESEARCHER -> stringProvider.getString(R.string.system_prompt_academic_researcher)
            AiMode.PROFESSIONAL_STRATEGIST -> stringProvider.getString(R.string.system_prompt_professional_strategist)
        }

        val userPrompt = when (action) {
            AiAssistantAction.GIVE_IDEA -> stringProvider.getString(R.string.prompt_assistant_give_idea, noteName)
            AiAssistantAction.CONTINUE_WRITING -> stringProvider.getString(R.string.prompt_assistant_continue_writing, noteName, noteDescription)
            AiAssistantAction.CHANGE_PERSPECTIVE -> stringProvider.getString(R.string.prompt_assistant_change_perspective, noteName, noteDescription)
            AiAssistantAction.PROS_AND_CONS -> stringProvider.getString(R.string.prompt_assistant_pros_and_cons, noteDescription)
            AiAssistantAction.CREATE_TODO_LIST -> stringProvider.getString(R.string.prompt_assistant_create_todo, noteDescription)
            AiAssistantAction.SIMPLIFY -> stringProvider.getString(R.string.prompt_assistant_simplify, noteDescription)
            AiAssistantAction.SUGGEST_A_TITLE -> stringProvider.getString(R.string.prompt_assistant_suggest_title, noteDescription)
        }

        val url = getChatCompletionsUrl()
        val request = ChatCompletionRequest(
            model = modelName,
            stream = true,
            messages = listOf(
                ChatMessageDto(role = "system", content = systemPrompt),
                ChatMessageDto(role = "user", content = userPrompt)
            ),
            temperature = temperature
        )

        streamChatCompletion(url, apiKey, request)
    }.catch {
        emit(stringProvider.getString(R.string.error_api_request_failed, it.message ?: "Unknown error"))
    }

    fun generateChatResponse(
        history: List<ChatMessage>,
        apiKey: String,
        aiMode: AiMode = AiMode.NOTE_ASSISTANT,
        mentionedNote: Note? = null,
        imageBase64: String? = null
    ): Flow<String> = flow {
        if (apiKey.isBlank()) {
            throw ApiKeyMissingException(stringProvider.getString(R.string.error_no_user_api_key))
        }

        val modelName = settingsRepository.settings.first().selectedModelName
        val temperature = if (aiMode == AiMode.CREATIVE_MIND) CREATIVE_TEMPERATURE else DEFAULT_TEMPERATURE

        val baseSystemPrompt = when (aiMode) {
            AiMode.NOTE_ASSISTANT -> stringProvider.getString(R.string.system_prompt_note_assistant)
            AiMode.CREATIVE_MIND -> stringProvider.getString(R.string.system_prompt_creative_mind)
            AiMode.ACADEMIC_RESEARCHER -> stringProvider.getString(R.string.system_prompt_academic_researcher)
            AiMode.PROFESSIONAL_STRATEGIST -> stringProvider.getString(R.string.system_prompt_professional_strategist)
        }

        val messages = mutableListOf<ChatMessageDto>()
        messages.add(ChatMessageDto(role = "system", content = baseSystemPrompt))

        if (mentionedNote != null) {
            val noteContextPrompt = stringProvider.getString(
                R.string.prompt_mention_context,
                mentionedNote.name,
                mentionedNote.description
            )
            messages.add(ChatMessageDto(role = "user", content = noteContextPrompt))
            val ackPrompt = stringProvider.getString(R.string.prompt_mention_ack, mentionedNote.name)
            messages.add(ChatMessageDto(role = "assistant", content = ackPrompt))
        }

        history.filter { it.participant != Participant.ERROR && !it.isLoading }
            .forEach { msg ->
                messages.add(ChatMessageDto(
                    role = if (msg.participant == Participant.USER) "user" else "assistant",
                    content = msg.text
                ))
            }

        val url = getChatCompletionsUrl()
        val request = ChatCompletionRequest(
            model = modelName,
            stream = true,
            messages = messages,
            temperature = temperature
        )

        streamChatCompletion(url, apiKey, request)
    }.catch {
        emit(stringProvider.getString(R.string.error_api_request_failed, it.message ?: "Unknown error"))
    }

    suspend fun generateDraft(topic: String, apiKey: String, aiMode: AiMode = AiMode.NOTE_ASSISTANT): Result<String> {
        if (apiKey.isBlank()) {
            return Result.failure(ApiKeyMissingException(stringProvider.getString(R.string.error_no_user_api_key)))
        }

        val modelName = settingsRepository.settings.first().selectedModelName
        val temperature = if (aiMode == AiMode.CREATIVE_MIND) CREATIVE_TEMPERATURE else DEFAULT_TEMPERATURE

        val systemPrompt = when (aiMode) {
            AiMode.NOTE_ASSISTANT -> stringProvider.getString(R.string.system_prompt_note_assistant)
            AiMode.CREATIVE_MIND -> stringProvider.getString(R.string.system_prompt_creative_mind)
            AiMode.ACADEMIC_RESEARCHER -> stringProvider.getString(R.string.system_prompt_academic_researcher)
            AiMode.PROFESSIONAL_STRATEGIST -> stringProvider.getString(R.string.system_prompt_professional_strategist)
        }
        val userPrompt = stringProvider.getString(R.string.prompt_assistant_draft_anything, topic)

        val url = getChatCompletionsUrl()
        val request = ChatCompletionRequest(
            model = modelName,
            stream = false,
            messages = listOf(
                ChatMessageDto(role = "system", content = systemPrompt),
                ChatMessageDto(role = "user", content = userPrompt)
            ),
            temperature = temperature
        )

        return try {
            val response = httpClient.post(url) {
                header("Authorization", "Bearer $apiKey")
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            val body = response.bodyAsChannel().readUTF8Line() ?: ""
            val completion = json.decodeFromString<ChatCompletionResponse>(body)
            val text = completion.choices.firstOrNull()?.message?.content
            if (text != null) {
                Result.success(text)
            } else {
                Result.failure(Exception("Empty response from API."))
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Result.failure(e)
        }
    }

    suspend fun generateDraftFromAttachment(
        prompt: String,
        imageBase64: String,
        apiKey: String,
        aiMode: AiMode
    ): Result<String> {
        if (apiKey.isBlank()) {
            return Result.failure(ApiKeyMissingException(stringProvider.getString(R.string.error_no_user_api_key)))
        }

        val modelName = settingsRepository.settings.first().selectedModelName
        val temperature = if (aiMode == AiMode.CREATIVE_MIND) CREATIVE_TEMPERATURE else DEFAULT_TEMPERATURE

        val systemPrompt = when (aiMode) {
            AiMode.NOTE_ASSISTANT -> stringProvider.getString(R.string.system_prompt_note_assistant)
            AiMode.CREATIVE_MIND -> stringProvider.getString(R.string.system_prompt_creative_mind)
            AiMode.ACADEMIC_RESEARCHER -> stringProvider.getString(R.string.system_prompt_academic_researcher)
            AiMode.PROFESSIONAL_STRATEGIST -> stringProvider.getString(R.string.system_prompt_professional_strategist)
        }

        val userPrompt = """
        User prompt: "$prompt"

        Analyze the attached image and create a note draft based on this prompt.
        You MUST respond in this format:
        TITLE: [write title here]

        CONTENT: [write note content here]
        """.trimIndent()

        val url = getChatCompletionsUrl()
        val request = ChatCompletionRequest(
            model = modelName,
            stream = false,
            messages = listOf(
                ChatMessageDto(role = "system", content = systemPrompt),
                ChatMessageDto(role = "user", content = userPrompt)
            ),
            temperature = temperature
        )

        return try {
            val response = httpClient.post(url) {
                header("Authorization", "Bearer $apiKey")
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            val body = response.bodyAsChannel().readUTF8Line() ?: ""
            val completion = json.decodeFromString<ChatCompletionResponse>(body)
            val text = completion.choices.firstOrNull()?.message?.content
            if (text != null) {
                Result.success(text)
            } else {
                Result.failure(Exception("Empty response from API."))
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Result.failure(e)
        }
    }

    suspend fun generateChatOrCommandResponse(
        noteContext: String,
        userRequest: String,
        chatHistory: List<ChatMessage>,
        apiKey: String
    ): Result<String> = withContext(Dispatchers.IO) {
        if (apiKey.isBlank()) {
            return@withContext Result.failure(ApiKeyMissingException(stringProvider.getString(R.string.error_no_user_api_key)))
        }

        try {
            val modelName = settingsRepository.settings.first().selectedModelName

            val systemPrompt = buildString {
                append(stringProvider.getString(R.string.system_prompt_command_chat_header))
                append("\n\n")
                append(stringProvider.getString(R.string.system_prompt_intent_chat_desc))
                append("\n\n")
                append(stringProvider.getString(R.string.system_prompt_intent_edit_desc))
                append("\n\n")
                append(stringProvider.getString(R.string.system_prompt_json_instruction))
            }

            val messages = mutableListOf<ChatMessageDto>()
            messages.add(ChatMessageDto(role = "system", content = systemPrompt))
            messages.add(ChatMessageDto(role = "system", content = stringProvider.getString(R.string.system_prompt_context_header, noteContext)))

            chatHistory.filter { !it.isLoading && it.participant != Participant.ERROR }
                .forEach { msg ->
                    messages.add(ChatMessageDto(
                        role = if (msg.participant == Participant.USER) "user" else "assistant",
                        content = msg.text
                    ))
                }

            val finalRequestPrompt = stringProvider.getString(R.string.system_prompt_request_header, userRequest)
            messages.add(ChatMessageDto(role = "user", content = finalRequestPrompt))

            val url = getChatCompletionsUrl()
            val request = ChatCompletionRequest(
                model = modelName,
                stream = false,
                messages = messages,
                temperature = 0.4
            )

            val response = httpClient.post(url) {
                header("Authorization", "Bearer $apiKey")
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            val body = response.bodyAsChannel().readUTF8Line() ?: ""
            val completion = json.decodeFromString<ChatCompletionResponse>(body)
            val responseText = completion.choices.firstOrNull()?.message?.content

            if (responseText.isNullOrBlank()) {
                Result.failure(Exception("Empty response from API."))
            } else {
                val cleanJson = responseText.trim()
                    .removePrefix("```json")
                    .removePrefix("```")
                    .removeSuffix("```")
                    .trim()

                Result.success(cleanJson)
            }

        } catch (e: Exception) {
            e.printStackTrace()
            Result.failure(e)
        }
    }

    suspend fun generateActionPlan(
        userRequest: String,
        chatHistory: List<ChatMessage>,
        apiKey: String,
        aiMode: AiMode,
        mentionedNote: Note?
    ): Result<String> = withContext(Dispatchers.IO) {

        if (apiKey.isBlank()) {
            return@withContext Result.failure(ApiKeyMissingException(stringProvider.getString(R.string.error_no_user_api_key)))
        }

        try {
            val modelName = settingsRepository.settings.first().selectedModelName
            val temperature = if (aiMode == AiMode.CREATIVE_MIND) 0.5 else 0.2

            val systemPrompt = stringProvider.getString(R.string.system_prompt_automation_engine)
            val messages = mutableListOf<ChatMessageDto>()
            messages.add(ChatMessageDto(role = "system", content = systemPrompt))

            if (mentionedNote != null) {
                val noteContextPrompt = """
                --- CURRENT NOTE CONTEXT ---
                Title: "${mentionedNote.name}"
                Content: "${mentionedNote.description}"
            """.trimIndent()
                messages.add(ChatMessageDto(role = "user", content = noteContextPrompt))
                messages.add(ChatMessageDto(role = "assistant", content = "OK. Context for note '${mentionedNote.name}' has been loaded."))
            }

            chatHistory.filter { !it.isLoading && it.participant != Participant.ERROR }
                .forEach { msg ->
                    messages.add(ChatMessageDto(
                        role = if (msg.participant == Participant.USER) "user" else "assistant",
                        content = msg.text
                    ))
                }

            messages.add(ChatMessageDto(role = "user", content = userRequest))

            val url = getChatCompletionsUrl()
            val request = ChatCompletionRequest(
                model = modelName,
                stream = false,
                messages = messages,
                temperature = temperature
            )

            val response = httpClient.post(url) {
                header("Authorization", "Bearer $apiKey")
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            val body = response.bodyAsChannel().readUTF8Line() ?: ""
            val completion = json.decodeFromString<ChatCompletionResponse>(body)
            val responseText = completion.choices.firstOrNull()?.message?.content

            if (responseText.isNullOrBlank()) {
                Result.failure(Exception("Empty response from AI."))
            } else {
                val cleanJson = responseText.trim()
                    .removePrefix("```json")
                    .removePrefix("```")
                    .removeSuffix("```")
                    .trim()

                Log.d("LlmRepository", "Raw JSON Plan: $cleanJson")
                Result.success(cleanJson)
            }

        } catch (e: Exception) {
            e.printStackTrace()
            Result.failure(e)
        }
    }

    // --- Helper: stream a chat completion and emit text chunks ---
    private suspend fun kotlinx.coroutines.flow.FlowCollector<String>.streamChatCompletion(
        url: String,
        apiKey: String,
        request: ChatCompletionRequest
    ) {
        val response = httpClient.post(url) {
            header("Authorization", "Bearer $apiKey")
            contentType(ContentType.Application.Json)
            setBody(request)
        }

        val channel = response.bodyAsChannel()

        while (!channel.isClosedForRead) {
            val line = channel.readUTF8Line()
            if (line.isNullOrBlank()) continue

            if (line.startsWith("data:")) {
                val jsonData = line.substring(5).trim()
                if (jsonData == "[DONE]") break

                try {
                    val chunk = json.decodeFromString<ChatCompletionChunk>(jsonData)
                    val content = chunk.choices.firstOrNull()?.delta?.content
                    if (content != null) {
                        emit(content)
                    }
                } catch (e: Exception) {
                    Log.w("LlmStreamParse", "JSON parse error: $jsonData", e)
                }
            }
        }
    }

    val supportedLanguages = mapOf(
        TranslateLanguage.ENGLISH to "English",
        TranslateLanguage.TURKISH to "Türkçe",
        TranslateLanguage.GERMAN to "Deutsch",
        TranslateLanguage.FRENCH to "Français",
        TranslateLanguage.SPANISH to "Español",
        TranslateLanguage.ITALIAN to "Italiano",
        TranslateLanguage.JAPANESE to "日本語",
        TranslateLanguage.RUSSIAN to "Русский"
    )

    suspend fun getDownloadedModels(): Result<Set<String>> = withContext(Dispatchers.IO) {
        suspendCoroutine { continuation ->
            RemoteModelManager.getInstance().getDownloadedModels(TranslateRemoteModel::class.java)
                .addOnSuccessListener { models ->
                    continuation.resume(Result.success(models.map { it.language }.toSet()))
                }
                .addOnFailureListener {
                    continuation.resume(Result.failure(it))
                }
        }
    }

    suspend fun downloadLanguageModel(languageCode: String): Result<Unit> = withContext(Dispatchers.IO) {
        suspendCoroutine { continuation ->
            val model = TranslateRemoteModel.Builder(languageCode).build()
            val conditions = DownloadConditions.Builder().requireWifi().build()
            RemoteModelManager.getInstance().download(model, conditions)
                .addOnSuccessListener {
                    continuation.resume(Result.success(Unit))
                }
                .addOnFailureListener {
                    continuation.resume(Result.failure(it))
                }
        }
    }
    suspend fun deleteLanguageModel(languageCode: String): Result<Unit> = withContext(Dispatchers.IO) {
        suspendCoroutine { continuation ->
            val model = TranslateRemoteModel.Builder(languageCode).build()
            RemoteModelManager.getInstance().deleteDownloadedModel(model)
                .addOnSuccessListener {
                    continuation.resume(Result.success(Unit))
                }
                .addOnFailureListener {
                    continuation.resume(Result.failure(it))
                }
        }
    }

    private suspend fun identifyLanguage(text: String): Result<String> = suspendCoroutine { continuation ->
        val languageIdentifier = LanguageIdentification.getClient()
        languageIdentifier.identifyLanguage(text)
            .addOnSuccessListener { languageCode ->
                if (languageCode == "und") {
                    continuation.resume(Result.failure(Exception("The source language could not be identified.")))
                } else {
                    continuation.resume(Result.success(languageCode))
                }
                languageIdentifier.close()
            }
            .addOnFailureListener {
                continuation.resume(Result.failure(it))
                languageIdentifier.close()
            }
    }

    suspend fun translateOnDevice(text: String, targetLanguage: String): Result<String> = withContext(Dispatchers.IO) {
        if (text.isBlank()) return@withContext Result.success("")

        val langIdResult = identifyLanguage(text)
        if (langIdResult.isFailure) {
            return@withContext Result.failure(langIdResult.exceptionOrNull()!!)
        }
        val sourceLanguageCode = langIdResult.getOrNull()!!

        if (sourceLanguageCode.equals(targetLanguage, ignoreCase = true)) {
            return@withContext Result.failure(Exception("The note text is already in the language you want to translate."))
        }

        val options = TranslatorOptions.Builder()
            .setSourceLanguage(sourceLanguageCode)
            .setTargetLanguage(targetLanguage)
            .build()
        val translator = Translation.getClient(options)

        try {
            val downloadResult = suspendCoroutine<Result<Unit>> { continuation ->
                translator.downloadModelIfNeeded()
                    .addOnSuccessListener { continuation.resume(Result.success(Unit)) }
                    .addOnFailureListener { continuation.resume(Result.failure(it)) }
            }
            if (downloadResult.isFailure) {
                throw downloadResult.exceptionOrNull()!!
            }

            val originalLines = text.split('\n')
            val translatedLines = mutableListOf<String>()

            for (line in originalLines) {
                if (line.isBlank()) {
                    translatedLines.add(line)
                } else {
                    val translatedLineResult = suspendCoroutine<Result<String>> { continuation ->
                        translator.translate(line)
                            .addOnSuccessListener { translatedText -> continuation.resume(Result.success(translatedText)) }
                            .addOnFailureListener { exception -> continuation.resume(Result.failure(exception)) }
                    }

                    if (translatedLineResult.isSuccess) {
                        translatedLines.add(translatedLineResult.getOrThrow())
                    } else {
                        throw translatedLineResult.exceptionOrNull()!!
                    }
                }
            }
            Result.success(translatedLines.joinToString("\n"))

        } catch (e: Exception) {
            Log.e("TranslateError", "Translation error", e)
            Result.failure(e)
        } finally {
            translator.close()
        }
    }
}
