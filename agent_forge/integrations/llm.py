from typing import Dict, Any, Optional, List
import os
from loguru import logger
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, template: str, inputs: Dict[str, Any]) -> str:
        """Generate text based on a template and inputs."""
        pass
    
    @abstractmethod
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response based on conversation history."""
        pass


    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider with configuration."""
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = config.get("base_url") or os.getenv("BASE_URL")
        self.model = config.get("model") or os.getenv("PRIMARY_MODEL") or "gpt-4"
        self.temperature = config.get("temperature", 0.7)
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable.")
    
    def generate(self, template: str, inputs: Dict[str, Any]) -> str:
        """Generate text using OpenAI."""
        try:
            import openai
            client_args = {"api_key": self.api_key}
            
            # Add base_url if specified
            if self.base_url:
                client_args["base_url"] = self.base_url
                
            client = openai.OpenAI(**client_args)
            
            # Format the template with inputs
            prompt = template
            for key, value in inputs.items():
                placeholder = f"{{{key}}}"
                prompt = prompt.replace(placeholder, str(value))
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are an AI agent builder assistant."},
                          {"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            return "ERROR: OpenAI package not installed."
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response based on conversation history."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            return f"ERROR: {str(e)}"

    def provider_name(self) -> str:
        return "OpenAI"
    
    def available_models(self) -> List[str]:
        """Return list of available models from this provider."""
        models = [
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
        
        # Add custom model if set
        custom_model = self.model
        if custom_model and custom_model not in models:
            models.append(custom_model)
            
        return models

class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Anthropic provider with configuration."""
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.model = config.get("model", "claude-2")
        self.temperature = config.get("temperature", 0.7)
        
        if not self.api_key:
            logger.warning("Anthropic API key not provided. Please set ANTHROPIC_API_KEY environment variable.")
    
    def generate(self, template: str, inputs: Dict[str, Any]) -> str:
        """Generate text using Anthropic."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Format the template with inputs
            prompt = template
            for key, value in inputs.items():
                placeholder = f"{{{key}}}"
                prompt = prompt.replace(placeholder, str(value))
            
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            return response.content[0].text
        except ImportError:
            logger.error("Anthropic package not installed. Install with: pip install anthropic")
            return "ERROR: Anthropic package not installed."
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response based on conversation history."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Convert to Anthropic format
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=anthropic_messages,
                temperature=self.temperature
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {str(e)}")
            return f"ERROR: {str(e)}"

    def provider_name(self) -> str:
        return "Anthropic"
    
    def available_models(self) -> List[str]:
        """Return list of available models from this provider."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]

class GoogleProvider(BaseLLMProvider):
    """Google AI (Gemini) LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Google provider with configuration."""
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        self.model = config.get("model", "gemini-pro")
        self.temperature = config.get("temperature", 0.7)
        
        if not self.api_key:
            logger.warning("Google API key not provided. Please set GOOGLE_API_KEY environment variable.")
    
    def provider_name(self) -> str:
        return "Google (Gemini)"
    
    def available_models(self) -> List[str]:
        """Return list of available models from this provider."""
        return [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-ultra"
        ]
    
    def generate(self, template: str, inputs: Dict[str, Any]) -> str:
        """Generate text using Google's Gemini models."""
        try:
            import google.generativeai as genai
            
            # Configure API key
            genai.configure(api_key=self.api_key)
            
            # Format the template with inputs
            prompt = template
            for key, value in inputs.items():
                placeholder = f"{{{key}}}"
                prompt = prompt.replace(placeholder, str(value))
            
            # Create generation config
            generation_config = {
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 0,
                "max_output_tokens": 2048,
            }
            
            # Initialize model
            model = genai.GenerativeModel(model_name=self.model, 
                                         generation_config=generation_config)
            
            # Generate response
            response = model.generate_content(prompt)
            
            # Return the response text
            return response.text
        except ImportError:
            logger.error("Google generative AI package not installed. Install with: pip install google-generativeai")
            return "ERROR: Google generativeai package not installed."
        except Exception as e:
            logger.error(f"Error generating text with Google: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response based on conversation history."""
        try:
            import google.generativeai as genai
            
            # Configure API key
            genai.configure(api_key=self.api_key)
            
            # Convert messages to Google's format
            chat_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                chat_messages.append({"role": role, "parts": [msg["content"]]})
            
            # Initialize model and start chat
            model = genai.GenerativeModel(model_name=self.model)
            chat = model.start_chat(history=chat_messages[:-1])
            
            # Send the last message and get response
            response = chat.send_message(chat_messages[-1]["parts"][0])
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Google: {str(e)}")
            return f"ERROR: {str(e)}"

class CohereLLMProvider(BaseLLMProvider):
    """Cohere LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Cohere provider with configuration."""
        self.api_key = config.get("api_key") or os.getenv("COHERE_API_KEY")
        self.model = config.get("model", "command")
        self.temperature = config.get("temperature", 0.7)
        
        if not self.api_key:
            logger.warning("Cohere API key not provided. Please set COHERE_API_KEY environment variable.")
    
    def provider_name(self) -> str:
        return "Cohere"
    
    def available_models(self) -> List[str]:
        """Return list of available models from this provider."""
        return [
            "command",
            "command-light",
            "command-nightly",
            "command-r",
            "command-r-plus"
        ]
    
    def generate(self, template: str, inputs: Dict[str, Any]) -> str:
        """Generate text using Cohere."""
        try:
            import cohere
            
            # Initialize client
            co = cohere.Client(api_key=self.api_key)
            
            # Format the template with inputs
            prompt = template
            for key, value in inputs.items():
                placeholder = f"{{{key}}}"
                prompt = prompt.replace(placeholder, str(value))
            
            # Generate response
            response = co.generate(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=2048
            )
            
            return response.generations[0].text
        except ImportError:
            logger.error("Cohere package not installed. Install with: pip install cohere")
            return "ERROR: Cohere package not installed."
        except Exception as e:
            logger.error(f"Error generating text with Cohere: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response based on conversation history."""
        try:
            import cohere
            
            # Initialize client
            co = cohere.Client(api_key=self.api_key)
            
            # Format the history into a chat transcript
            chat_history = []
            for msg in messages[:-1]:  # All except the last message
                role = "USER" if msg["role"] == "user" else "CHATBOT"
                chat_history.append({"role": role, "message": msg["content"]})
            
            # Get the last user message
            last_message = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
            
            # Generate response
            response = co.chat(
                model=self.model,
                message=last_message,
                chat_history=chat_history,
                temperature=self.temperature
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Cohere: {str(e)}")
            return f"ERROR: {str(e)}"

class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama provider with configuration."""
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llama2")
        self.temperature = config.get("temperature", 0.7)
        
        # Test connection
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                self.available_models_list = [model["name"] for model in response.json().get("models", [])]
                logger.info(f"Connected to Ollama with {len(self.available_models_list)} available models")
            else:
                logger.warning(f"Couldn't connect to Ollama server at {self.base_url}")
                self.available_models_list = []
        except Exception as e:
            logger.warning(f"Couldn't connect to Ollama server: {str(e)}")
            self.available_models_list = []
    
    def provider_name(self) -> str:
        return "Ollama (Local)"
    
    def available_models(self) -> List[str]:
        """Return list of available models from Ollama."""
        return self.available_models_list or ["llama2", "mistral", "vicuna"]
    
    def generate(self, template: str, inputs: Dict[str, Any]) -> str:
        """Generate text using Ollama local models."""
        try:
            import requests
            import json
            
            # Format the template with inputs
            prompt = template
            for key, value in inputs.items():
                placeholder = f"{{{key}}}"
                prompt = prompt.replace(placeholder, str(value))
            
            # Set up the request
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            }
            
            # Send the request
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                error_msg = f"Error from Ollama: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
                
        except ImportError:
            logger.error("Requests package not installed. Install with: pip install requests")
            return "ERROR: Requests package not installed."
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response based on conversation history using Ollama."""
        try:
            import requests
            
            # Format the conversation history into a single prompt
            conversation = ""
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation += f"{role}: {msg['content']}\n\n"
            
            conversation += "Assistant: "
            
            # Set up the request
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": conversation,
                "temperature": self.temperature,
                "stream": False
            }
            
            # Send the request
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                error_msg = f"Error from Ollama: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {str(e)}")
            return f"ERROR: {str(e)}"

class CustomProvider(BaseLLMProvider):
    """Custom LLM provider implementation for any model API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Custom provider with configuration."""
        self.api_key = config.get("api_key") or os.getenv("LLM_API_KEY")
        self.base_url = config.get("base_url") or os.getenv("CUSTOM_BASE_URL")
        self.model = config.get("model") or os.getenv("CUSTOM_MODEL")
        self.temperature = config.get("temperature", 0.7)
        self.headers = config.get("headers", {})
        self.api_type = config.get("api_type", "openai_compatible")
        
        # Add authorization header if API key provided
        if self.api_key and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
            
        if not self.base_url:
            logger.warning("Custom provider requires base_url. Please set in configuration or CUSTOM_BASE_URL environment variable.")
    
    def provider_name(self) -> str:
        return "Custom Model"
    
    def available_models(self) -> List[str]:
        """Return list of available models from this provider."""
        return [self.model] if self.model else ["custom-model"]
    
    def generate(self, template: str, inputs: Dict[str, Any]) -> str:
        """Generate text using custom API."""
        if not self.base_url:
            return "ERROR: Custom provider requires base_url"
            
        try:
            import requests
            import json
            
            # Format the template with inputs
            prompt = template
            for key, value in inputs.items():
                placeholder = f"{{{key}}}"
                prompt = prompt.replace(placeholder, str(value))
            
            # Build request based on API type
            if self.api_type == "openai_compatible":
                # Format as OpenAI API request
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an AI agent builder assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                    
            elif self.api_type == "raw_completion":
                # Simple request/response format
                payload = {
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "model": self.model
                }
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "") or data.get("output", "") or data.get("text", "")
            
            # Failed to get proper response
            logger.error(f"Error from custom API: {response.status_code}, {response.text}")
            return f"ERROR: Custom API returned status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error generating text with custom provider: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response based on conversation history using custom API."""
        if not self.base_url:
            return "ERROR: Custom provider requires base_url"
            
        try:
            import requests
            import json
            
            # Build request based on API type
            if self.api_type == "openai_compatible":
                # Format as OpenAI API request
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
            
            elif self.api_type == "raw_completion":
                # Format history into a prompt
                conversation = ""
                for msg in messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation += f"{role}: {msg['content']}\n\n"
                
                conversation += "Assistant: "
                
                payload = {
                    "prompt": conversation,
                    "temperature": self.temperature,
                    "model": self.model
                }
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "") or data.get("output", "") or data.get("text", "")
            
            # Failed to get proper response
            logger.error(f"Error from custom API: {response.status_code}, {response.text}")
            return f"ERROR: Custom API returned status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error generating text with custom provider: {str(e)}")
            return f"ERROR: {str(e)}"

class LLMProvider:
    """Main LLM provider that selects and manages different backend providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM provider with configuration."""
        self.config = config
        self.provider_name = config.get("provider", "openai").lower()
        self.provider = self._initialize_provider()
    
    def _initialize_provider(self) -> BaseLLMProvider:
        """Initialize the correct provider based on configuration."""
        provider_config = self.config.get(self.provider_name, {})
        
        if self.provider_name == "openai":
            return OpenAIProvider(provider_config)
        elif self.provider_name == "anthropic":
            return AnthropicProvider(provider_config)
        elif self.provider_name == "google":
            return GoogleProvider(provider_config)
        elif self.provider_name == "cohere":
            return CohereLLMProvider(provider_config)
        elif self.provider_name == "ollama":
            return OllamaProvider(provider_config)
        elif self.provider_name == "custom":
            return CustomProvider(provider_config)
        else:
            logger.warning(f"Unknown provider '{self.provider_name}', falling back to OpenAI")
            return OpenAIProvider(provider_config)
    
    def generate(self, template: str, inputs: Dict[str, Any]) -> str:
        """Generate text using the selected provider."""
        return self.provider.generate(template, inputs)
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response based on conversation history."""
        return self.provider.generate_with_history(messages)
        
    def list_available_providers(self) -> List[str]:
        """Get list of all available provider names."""
        return ["openai", "anthropic", "google", "cohere", "ollama", "custom"]
    
    def get_provider_models(self, provider_name: str = None) -> List[str]:
        """Get list of available models for a specific provider."""
        if provider_name is None or provider_name == self.provider_name:
            return self.provider.available_models()
            
        # Initialize the requested provider temporarily to get its models
        provider_config = self.config.get(provider_name, {})
        if provider_name == "openai":
            return OpenAIProvider(provider_config).available_models()
        elif provider_name == "anthropic":
            return AnthropicProvider(provider_config).available_models()
        elif provider_name == "google":
            return GoogleProvider(provider_config).available_models()
        elif provider_name == "cohere":
            return CohereLLMProvider(provider_config).available_models()
        elif provider_name == "ollama":
            return OllamaProvider(provider_config).available_models()
        else:
            return []
