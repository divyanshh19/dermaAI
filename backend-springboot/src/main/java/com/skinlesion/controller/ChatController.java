package com.skinlesion.controller;

import com.skinlesion.dto.*;
import com.skinlesion.service.ChatService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
@Tag(name = "AI Medical Chatbot", description = "Endpoints for NVIDIA Nemotron-powered medical assistant and conversation history management")
public class ChatController {

    private final ChatService chatService;

    @PostMapping
    @Operation(summary = "Send message to AI Medical Assistant", description = "Sends user prompt and optional prediction context to NVIDIA Nemotron microservice, preserving conversation history in MySQL.")
    public ResponseEntity<ChatResponse> chat(@Valid @RequestBody ChatRequest request) {
        return ResponseEntity.ok(chatService.processChat(request));
    }

    @PostMapping("/conversations")
    @Operation(summary = "Create a new chat conversation session")
    public ResponseEntity<ConversationResponse> createConversation() {
        return ResponseEntity.ok(chatService.createConversation());
    }

    @GetMapping("/conversations")
    @Operation(summary = "Retrieve all chat conversations")
    public ResponseEntity<List<ConversationResponse>> getAllConversations() {
        return ResponseEntity.ok(chatService.getAllConversations());
    }

    @GetMapping("/conversations/{id}")
    @Operation(summary = "Get a specific conversation and its chat history")
    public ResponseEntity<ConversationResponse> getConversationById(@PathVariable("id") String id) {
        return ResponseEntity.ok(chatService.getConversationById(id));
    }

    @DeleteMapping("/conversations/{id}")
    @Operation(summary = "Delete a conversation and its messages")
    public ResponseEntity<Void> deleteConversation(@PathVariable("id") String id) {
        chatService.deleteConversation(id);
        return ResponseEntity.noContent().build();
    }
}
