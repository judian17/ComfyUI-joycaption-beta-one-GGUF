import torch
from PIL import Image
import folder_paths # ComfyUI utility
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
import io
import sys # For suppressing/capturing stdout/stderr
from torchvision.transforms import ToPILImage
import gc # Import the garbage collection module

# Constants for caption generation, copied from original nodes.py
CAPTION_TYPE_MAP = {
	"Descriptive": [
		"Write a detailed description for this image.",
		"Write a detailed description for this image in {word_count} words or less.",
		"Write a {length} detailed description for this image.",
	],
	"Descriptive (Casual)": [
		"Write a descriptive caption for this image in a casual tone.",
		"Write a descriptive caption for this image in a casual tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a casual tone.",
	],
	"Straightforward": [
		"Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
	],
	"Stable Diffusion Prompt": [
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
		"Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
	],
	"MidJourney": [
		"Write a MidJourney prompt for this image.",
		"Write a MidJourney prompt for this image within {word_count} words.",
		"Write a {length} MidJourney prompt for this image.",
	],
	"Danbooru tag list": [
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
	],
	"e621 tag list": [
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
	],
	"Rule34 tag list": [
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
	],
	"Booru-like tag list": [
		"Write a list of Booru-like tags for this image.",
		"Write a list of Booru-like tags for this image within {word_count} words.",
		"Write a {length} list of Booru-like tags for this image.",
	],
	"Art Critic": [
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
	],
	"Product Listing": [
		"Write a caption for this image as though it were a product listing.",
		"Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
		"Write a {length} caption for this image as though it were a product listing.",
	],
	"Social Media Post": [
		"Write a caption for this image as if it were being used for a social media post.",
		"Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
		"Write a {length} caption for this image as if it were being used for a social media post.",
	],
}
EXTRA_OPTIONS = [
	"", "If there is a person/character in the image you must refer to them as {name}.",
	"Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
	"Include information about lighting.", "Include information about camera angle.",
	"Include information about whether there is a watermark or not.", "Include information about whether there are JPEG artifacts or not.",
	"If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
	"Do NOT include anything sexual; keep it PG.", "Do NOT mention the image's resolution.",
	"You MUST include information about the subjective aesthetic quality of the image from low to very high.",
	"Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
	"Do NOT mention any text that is in the image.", "Specify the depth of field and whether the background is in focus or blurred.",
	"If applicable, mention the likely use of artificial or natural lighting sources.", "Do NOT use any ambiguous language.",
	"Include whether the image is sfw, suggestive, or nsfw.", "ONLY describe the most important elements of the image.",
	"If it is a work of art, do not include the artist's name or the title of the work.",
	"Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
	"""Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.""",
	"Do NOT use polite euphemisms—lean into blunt, casual phrasing.", "Include information about the ages of any people/characters when applicable.",
	"Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
	"Do not mention the mood/feeling/etc of the image.", "Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).",
	"If there is a watermark, you must mention it.",
	"""Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.""",
] # This EXTRA_OPTIONS list is no longer directly used by JoyCaptionGGUF node for dropdowns.
CAPTION_LENGTH_CHOICES = (["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)])

def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
	if caption_type not in CAPTION_TYPE_MAP:
		print(f"JoyCaption (GGUF) Warning: Unknown caption_type '{caption_type}'. Using default.")
		default_template_key = list(CAPTION_TYPE_MAP.keys())[0]
		prompt_templates = CAPTION_TYPE_MAP.get(caption_type, CAPTION_TYPE_MAP[default_template_key])
	else:
		prompt_templates = CAPTION_TYPE_MAP[caption_type]

	if caption_length == "any": map_idx = 0
	elif isinstance(caption_length, str) and caption_length.isdigit(): map_idx = 1
	else: map_idx = 2

	if map_idx >= len(prompt_templates): map_idx = 0

	prompt = prompt_templates[map_idx]

	# Format the prompt first to handle {name} if it's part of the base prompt template
	# (though typically {name} is expected in extra_options)
	try:
		prompt = prompt.format(name=name_input or "{NAME}", length=caption_length, word_count=caption_length)
	except KeyError as e:
		# If {name} is not in the base prompt, it might be in extra_options, so don't error out yet.
		# Or, if other keys are missing, this is a genuine error.
		if 'name' not in str(e).lower(): # Check if the error is specifically about 'name'
			print(f"JoyCaption (GGUF) Warning: Prompt template formatting error for caption_type '{caption_type}', map_idx {map_idx}. Missing key: {e}")
			# Return the unformatted prompt with an error message appended
			return prompt_templates[map_idx] + f" (Base prompt formatting error: missing key {e})"


	if extra_options:
		# Process extra options, some of which might contain {name}
		processed_extra_options = []
		for opt in extra_options:
			try:
				processed_extra_options.append(opt.format(name=name_input or "{NAME}"))
			except KeyError as e_opt:
				# If an extra option has a formatting key other than {name} that's missing, warn and use raw.
				if 'name' not in str(e_opt).lower():
					print(f"JoyCaption (GGUF) Warning: Extra option formatting error: '{opt}'. Missing key: {e_opt}")
					processed_extra_options.append(opt + f" (Extra option formatting error: missing key {e_opt})")
				else: # If it's just {name} missing and name_input is empty, it's fine.
					processed_extra_options.append(opt)


		prompt += " " + " ".join(processed_extra_options)

	# Final check for any remaining unformatted placeholders if name_input was crucial and not provided
	# This check is a bit broad, but aims to catch unreplaced {name}, {length}, {word_count}
	if "{name}" in prompt and not name_input:
		# This case should ideally be handled by `name=name_input or "{NAME}"`
		# but as a safeguard if prompt still contains it.
		pass # It's okay if {name} remains if no name was provided.
	if "{length}" in prompt and caption_length != "any" and not (isinstance(caption_length, str) and caption_length.isdigit()):
		print(f"JoyCaption (GGUF) Warning: Prompt template for '{caption_type}' might have unformatted '{{length}}'.")
	if "{word_count}" in prompt and not (isinstance(caption_length, str) and caption_length.isdigit()):
		print(f"JoyCaption (GGUF) Warning: Prompt template for '{caption_type}' might have unformatted '{{word_count}}'.")

	return prompt

def get_gguf_model_paths(subfolder="llava_gguf"):
    base_models_dir = Path(folder_paths.models_dir)
    models_path = base_models_dir / subfolder
    if not models_path.exists():
        try:
            models_path.mkdir(parents=True, exist_ok=True)
            print(f"JoyCaption (GGUF): Created directory {models_path}")
        except Exception as e:
            print(f"JoyCaption (GGUF): Failed to create directory {models_path}: {e}")
            return []
    return sorted([str(p.name) for p in models_path.glob("*.gguf")])

def get_mmproj_paths(subfolder="llava_gguf"):
    base_models_dir = Path(folder_paths.models_dir)
    models_path = base_models_dir / subfolder
    if not models_path.exists(): return []
    return sorted([str(p.name) for p in models_path.glob("*.gguf")] + [str(p.name) for p in models_path.glob("*.bin")])

class JoyCaptionPredictorGGUF:
    def __init__(self, model_name: str, mmproj_name: str, n_gpu_layers: int = 0, n_ctx: int = 2048, seed: int = -1, subfolder="llava_gguf"):
        self.llm = None
        self.chat_handler_exit_stack = None # Will store the ExitStack of the chat_handler

        base_models_dir = Path(folder_paths.models_dir)
        model_path_full = base_models_dir / subfolder / model_name
        mmproj_path_full = base_models_dir / subfolder / mmproj_name

        if not model_path_full.exists(): raise FileNotFoundError(f"GGUF Model file not found: {model_path_full}")
        if not mmproj_path_full.exists(): raise FileNotFoundError(f"mmproj file not found: {mmproj_path_full}")
        
        _chat_handler_for_llama = None # Temporary local var
        try:
            _chat_handler_for_llama = Llava15ChatHandler(clip_model_path=str(mmproj_path_full))
            if hasattr(_chat_handler_for_llama, '_exit_stack'):
                self.chat_handler_exit_stack = _chat_handler_for_llama._exit_stack
            else:
                print("JoyCaption (GGUF) Warning: Llava15ChatHandler does not have _exit_stack attribute.")

            self.llm = Llama(
                model_path=str(model_path_full),
                chat_handler=_chat_handler_for_llama, 
                n_ctx=n_ctx,
                logits_all=True,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                seed=seed,
            )
            print(f"JoyCaption (GGUF): Loaded model {model_name} with mmproj {mmproj_name}, seed {seed}.")
        except Exception as e:
            print(f"JoyCaption (GGUF): Error loading GGUF model: {e}")
            if self.chat_handler_exit_stack is not None:
                try:
                    print("JoyCaption (GGUF): Attempting to close chat_handler_exit_stack due to load error.")
                    self.chat_handler_exit_stack.close()
                except Exception as e_close:
                    print(f"JoyCaption (GGUF): Error closing chat_handler_exit_stack on load error: {e_close}")
            if self.llm is not None: # Should be None if Llama init failed, but as a safeguard
                del self.llm
            self.llm = None # Ensure llm is None
            self.chat_handler_exit_stack = None # Clear stack
            raise e
        
    @torch.inference_mode()
    def generate(self, image: Image.Image, system: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
        if self.llm is None: return "Error: GGUF model not loaded."

        buffered = io.BytesIO()
        image_format = image.format if image.format else "PNG"
        save_format = "JPEG" if image_format.upper() == "JPEG" else "PNG"
        image.save(buffered, format=save_format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_url = f"data:image/{save_format.lower()};base64,{img_base64}"

        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "content": prompt.strip()}]}
        ]
        
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        caption = ""
        try:
            response = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_new_tokens if max_new_tokens > 0 else None,
                temperature=temperature if temperature > 0 else 0.0, top_p=top_p, top_k=top_k if top_k > 0 else 0,
            )
            caption = response['choices'][0]['message']['content']
        except Exception as e:
            print(f"JoyCaption (GGUF): Error during GGUF model generation: {e}")
            return f"Error generating caption: {e}"
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        return caption.strip()

AVAILABLE_GGUF_MODELS = []
AVAILABLE_MMPROJ_FILES = []

def _populate_file_lists():
    global AVAILABLE_GGUF_MODELS, AVAILABLE_MMPROJ_FILES
    if not AVAILABLE_GGUF_MODELS: AVAILABLE_GGUF_MODELS = get_gguf_model_paths()
    if not AVAILABLE_MMPROJ_FILES: AVAILABLE_MMPROJ_FILES = get_mmproj_paths()
    if not AVAILABLE_GGUF_MODELS: AVAILABLE_GGUF_MODELS = ["None (place models in ComfyUI/models/llava_gguf)"]
    if not AVAILABLE_MMPROJ_FILES: AVAILABLE_MMPROJ_FILES = ["None (place mmproj files in ComfyUI/models/llava_gguf)"]

_populate_file_lists()

class JoyCaptionGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "image": ("IMAGE",), "gguf_model": (AVAILABLE_GGUF_MODELS,), "mmproj_file": (AVAILABLE_MMPROJ_FILES,),
            "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000}),
            "n_ctx": ("INT", {"default": 2048, "min": 512, "max": 8192}),
            "caption_type": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive (Casual)"}),
            "caption_length": (CAPTION_LENGTH_CHOICES,),
            "max_new_tokens": ("INT", {"default": 512, "min": 0, "max": 4096}),
            "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            "unload_after_generate": ("BOOLEAN", {"default": False}),
        }
        opt = {
            "extra_options_input": ("JJC_GGUF_EXTRA_OPTION",)
        }
        return {"required": req, "optional": opt}

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("STRING","STRING"), ("query", "caption"), "generate", "JoyCaption"

    def __init__(self):
        self.predictor_gguf = None
        self.current_model_key = None

    def generate(self, image, gguf_model, mmproj_file, n_gpu_layers, n_ctx, caption_type, caption_length,
                 max_new_tokens, temperature, top_p, top_k, seed, unload_after_generate, extra_options_input=None):
        if gguf_model.startswith("None") or mmproj_file.startswith("None"):
             return ("Error: GGUF model or mmproj file not selected/found.", "Please place models in ComfyUI/models/llava_gguf and select them.")

        model_key = (gguf_model, mmproj_file, n_gpu_layers, n_ctx, seed)

        if self.predictor_gguf is None or self.current_model_key != model_key:
            if self.predictor_gguf is not None:
                if hasattr(self.predictor_gguf, 'chat_handler_exit_stack') and self.predictor_gguf.chat_handler_exit_stack is not None:
                    try:
                        print("JoyCaption (GGUF): Manually closing chat_handler_exit_stack (model switch).")
                        self.predictor_gguf.chat_handler_exit_stack.close()
                    except Exception as e_close:
                        print(f"JoyCaption (GGUF): Error closing chat_handler_exit_stack (model switch): {e_close}")
                    self.predictor_gguf.chat_handler_exit_stack = None

                if hasattr(self.predictor_gguf, 'llm') and self.predictor_gguf.llm is not None:
                    del self.predictor_gguf.llm
                    self.predictor_gguf.llm = None
                
                del self.predictor_gguf
                self.predictor_gguf = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                print("JoyCaption (GGUF): Old model unloaded (model switch).")
            
            try:
                self.predictor_gguf = JoyCaptionPredictorGGUF(model_name=gguf_model, mmproj_name=mmproj_file, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
                self.current_model_key = model_key
            except Exception as e:
                print(f"JoyCaption (GGUF): Node-level error loading GGUF model: {e}")
                return (f"Error loading GGUF model: {e}", f"Details: {str(e)}")

        extras = []
        person_name_from_options = ""
        if extra_options_input:
            if isinstance(extra_options_input, tuple) and len(extra_options_input) == 2:
                extras, person_name_from_options = extra_options_input
                if not isinstance(extras, list): extras = []
                if not isinstance(person_name_from_options, str): person_name_from_options = ""
            else: # Should not happen if connected to JoyCaptionGGUFExtraOptions
                print(f"JoyCaption (GGUF) Warning: extra_options_input is not in the expected format (list, str). Received: {type(extra_options_input)}")


        prompt = build_prompt(caption_type, caption_length, extras, person_name_from_options)
        system_prompt = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."
        
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        response = self.predictor_gguf.generate(pil_image, system_prompt, prompt, max_new_tokens, temperature, top_p, top_k)
        
        if unload_after_generate and self.predictor_gguf is not None:
            if hasattr(self.predictor_gguf, 'chat_handler_exit_stack') and self.predictor_gguf.chat_handler_exit_stack is not None:
                try:
                    print("JoyCaption (GGUF): Manually closing chat_handler_exit_stack (unload_after_generate).")
                    self.predictor_gguf.chat_handler_exit_stack.close()
                except Exception as e_close:
                    print(f"JoyCaption (GGUF): Error closing chat_handler_exit_stack (unload_after_generate): {e_close}")
                self.predictor_gguf.chat_handler_exit_stack = None

            if hasattr(self.predictor_gguf, 'llm') and self.predictor_gguf.llm is not None:
                del self.predictor_gguf.llm
                self.predictor_gguf.llm = None # Explicitly set to None
            
            del self.predictor_gguf
            self.predictor_gguf = None
            self.current_model_key = None # Crucial to reset this
            gc.collect() 
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print("JoyCaption (GGUF): Model unloaded, chat_handler_exit_stack closed, GC run, CUDA cache emptied (unload_after_generate).")
            
        return (prompt, response)

class JoyCaptionGGUFExtraOptions:
    CATEGORY = 'JoyCaption'
    FUNCTION = "generate_options"
    RETURN_TYPES = ("JJC_GGUF_EXTRA_OPTION",)
    RETURN_NAMES = ("extra_options_gguf",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refer_character_name": ("BOOLEAN", {"default": False}),
                "exclude_people_info": ("BOOLEAN", {"default": False}),
                "include_lighting": ("BOOLEAN", {"default": False}),
                "include_camera_angle": ("BOOLEAN", {"default": False}),
                "include_watermark_info": ("BOOLEAN", {"default": False}), # Renamed from include_watermark to avoid conflict
                "include_JPEG_artifacts": ("BOOLEAN", {"default": False}),
                "include_exif": ("BOOLEAN", {"default": False}),
                "exclude_sexual": ("BOOLEAN", {"default": False}),
                "exclude_image_resolution": ("BOOLEAN", {"default": False}),
                "include_aesthetic_quality": ("BOOLEAN", {"default": False}),
                "include_composition_style": ("BOOLEAN", {"default": False}),
                "exclude_text": ("BOOLEAN", {"default": False}),
                "specify_depth_field": ("BOOLEAN", {"default": False}),
                "specify_lighting_sources": ("BOOLEAN", {"default": False}),
                "do_not_use_ambiguous_language": ("BOOLEAN", {"default": False}),
                "include_nsfw_rating": ("BOOLEAN", {"default": False}), # Renamed from include_nsfw
                "only_describe_most_important_elements": ("BOOLEAN", {"default": False}),
                "do_not_include_artist_name_or_title": ("BOOLEAN", {"default": False}),
                "identify_image_orientation": ("BOOLEAN", {"default": False}),
                "use_vulgar_slang_and_profanity": ("BOOLEAN", {"default": False}),
                "do_not_use_polite_euphemisms": ("BOOLEAN", {"default": False}),
                "include_character_age": ("BOOLEAN", {"default": False}),
                "include_camera_shot_type": ("BOOLEAN", {"default": False}),
                "exclude_mood_feeling": ("BOOLEAN", {"default": False}),
                "include_camera_vantage_height": ("BOOLEAN", {"default": False}),
                "mention_watermark_explicitly": ("BOOLEAN", {"default": False}), # Renamed from mention_watermark
                "avoid_meta_descriptive_phrases": ("BOOLEAN", {"default": False}),
                "character_name": ("STRING", {"default": "", "multiline": False, "placeholder": "e.g., 'Skywalker'"}),
            }
        }

    def generate_options(self, refer_character_name, exclude_people_info, include_lighting, include_camera_angle,
                         include_watermark_info, include_JPEG_artifacts, include_exif, exclude_sexual,
                         exclude_image_resolution, include_aesthetic_quality, include_composition_style,
                         exclude_text, specify_depth_field, specify_lighting_sources,
                         do_not_use_ambiguous_language, include_nsfw_rating, only_describe_most_important_elements,
                         do_not_include_artist_name_or_title, identify_image_orientation, use_vulgar_slang_and_profanity,
                         do_not_use_polite_euphemisms, include_character_age, include_camera_shot_type,
                         exclude_mood_feeling, include_camera_vantage_height, mention_watermark_explicitly,
                         avoid_meta_descriptive_phrases, character_name):

        extra_map = {
            "refer_character_name": "If there is a person/character in the image you must refer to them as {name}.",
            "exclude_people_info": "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
            "include_lighting": "Include information about lighting.",
            "include_camera_angle": "Include information about camera angle.",
            "include_watermark_info": "Include information about whether there is a watermark or not.",
            "include_JPEG_artifacts": "Include information about whether there are JPEG artifacts or not.",
            "include_exif": "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
            "exclude_sexual": "Do NOT include anything sexual; keep it PG.",
            "exclude_image_resolution": "Do NOT mention the image's resolution.",
            "include_aesthetic_quality": "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
            "include_composition_style": "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
            "exclude_text": "Do NOT mention any text that is in the image.",
            "specify_depth_field": "Specify the depth of field and whether the background is in focus or blurred.",
            "specify_lighting_sources": "If applicable, mention the likely use of artificial or natural lighting sources.",
            "do_not_use_ambiguous_language": "Do NOT use any ambiguous language.",
            "include_nsfw_rating": "Include whether the image is sfw, suggestive, or nsfw.",
            "only_describe_most_important_elements": "ONLY describe the most important elements of the image.",
            "do_not_include_artist_name_or_title": "If it is a work of art, do not include the artist's name or the title of the work.",
            "identify_image_orientation": "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
            "use_vulgar_slang_and_profanity": """Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.""",
            "do_not_use_polite_euphemisms": "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
            "include_character_age": "Include information about the ages of any people/characters when applicable.",
            "include_camera_shot_type": "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
            "exclude_mood_feeling": "Do not mention the mood/feeling/etc of the image.",
            "include_camera_vantage_height": "Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).",
            "mention_watermark_explicitly": "If there is a watermark, you must mention it.",
            "avoid_meta_descriptive_phrases": """Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc."""
        }
        
        selected_options = []
        # Iterate through the input arguments of this method (excluding self and character_name)
        # This is a bit manual; could use inspect if more dynamic behavior is needed.
        if refer_character_name: selected_options.append(extra_map["refer_character_name"])
        if exclude_people_info: selected_options.append(extra_map["exclude_people_info"])
        if include_lighting: selected_options.append(extra_map["include_lighting"])
        if include_camera_angle: selected_options.append(extra_map["include_camera_angle"])
        if include_watermark_info: selected_options.append(extra_map["include_watermark_info"])
        if include_JPEG_artifacts: selected_options.append(extra_map["include_JPEG_artifacts"])
        if include_exif: selected_options.append(extra_map["include_exif"])
        if exclude_sexual: selected_options.append(extra_map["exclude_sexual"])
        if exclude_image_resolution: selected_options.append(extra_map["exclude_image_resolution"])
        if include_aesthetic_quality: selected_options.append(extra_map["include_aesthetic_quality"])
        if include_composition_style: selected_options.append(extra_map["include_composition_style"])
        if exclude_text: selected_options.append(extra_map["exclude_text"])
        if specify_depth_field: selected_options.append(extra_map["specify_depth_field"])
        if specify_lighting_sources: selected_options.append(extra_map["specify_lighting_sources"])
        if do_not_use_ambiguous_language: selected_options.append(extra_map["do_not_use_ambiguous_language"])
        if include_nsfw_rating: selected_options.append(extra_map["include_nsfw_rating"])
        if only_describe_most_important_elements: selected_options.append(extra_map["only_describe_most_important_elements"])
        if do_not_include_artist_name_or_title: selected_options.append(extra_map["do_not_include_artist_name_or_title"])
        if identify_image_orientation: selected_options.append(extra_map["identify_image_orientation"])
        if use_vulgar_slang_and_profanity: selected_options.append(extra_map["use_vulgar_slang_and_profanity"])
        if do_not_use_polite_euphemisms: selected_options.append(extra_map["do_not_use_polite_euphemisms"])
        if include_character_age: selected_options.append(extra_map["include_character_age"])
        if include_camera_shot_type: selected_options.append(extra_map["include_camera_shot_type"])
        if exclude_mood_feeling: selected_options.append(extra_map["exclude_mood_feeling"])
        if include_camera_vantage_height: selected_options.append(extra_map["include_camera_vantage_height"])
        if mention_watermark_explicitly: selected_options.append(extra_map["mention_watermark_explicitly"])
        if avoid_meta_descriptive_phrases: selected_options.append(extra_map["avoid_meta_descriptive_phrases"])

        return ((selected_options, character_name or ""),)


class JoyCaptionCustomGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "image": ("IMAGE",), "gguf_model": (AVAILABLE_GGUF_MODELS,), "mmproj_file": (AVAILABLE_MMPROJ_FILES,),
            "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000}),
            "n_ctx": ("INT", {"default": 2048, "min": 512, "max": 8192}),
            "system_prompt": ("STRING", {"multiline": False, "default": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions." }),
            "user_query": ("STRING", {"multiline": True, "default": "Write a detailed description for this image." }),
            "max_new_tokens": ("INT", {"default": 512, "min": 0, "max": 4096}),
            "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            "unload_after_generate": ("BOOLEAN", {"default": False}),
        }
        opt = {
            "extra_options_input": ("JJC_GGUF_EXTRA_OPTION",)
        }
        return {"required": req, "optional": opt}

    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "generate", "JoyCaption"

    def __init__(self):
        self.predictor_gguf = None
        self.current_model_key = None

    def generate(self, image, gguf_model, mmproj_file, n_gpu_layers, n_ctx, system_prompt, user_query,
                 max_new_tokens, temperature, top_p, top_k, seed, unload_after_generate, extra_options_input=None):
        if gguf_model.startswith("None") or mmproj_file.startswith("None"):
             return ("Error: GGUF model or mmproj file not selected/found. Please place models in ComfyUI/models/llava_gguf and select them.",)

        model_key = (gguf_model, mmproj_file, n_gpu_layers, n_ctx, seed)

        if self.predictor_gguf is None or self.current_model_key != model_key:
            if self.predictor_gguf is not None:
                if hasattr(self.predictor_gguf, 'chat_handler_exit_stack') and self.predictor_gguf.chat_handler_exit_stack is not None:
                    try:
                        print("JoyCaption (GGUF): Manually closing chat_handler_exit_stack (model switch - custom node).")
                        self.predictor_gguf.chat_handler_exit_stack.close()
                    except Exception as e_close:
                        print(f"JoyCaption (GGUF): Error closing chat_handler_exit_stack (model switch - custom node): {e_close}")
                    self.predictor_gguf.chat_handler_exit_stack = None

                if hasattr(self.predictor_gguf, 'llm') and self.predictor_gguf.llm is not None:
                    del self.predictor_gguf.llm
                    self.predictor_gguf.llm = None # Explicitly set to None
                
                del self.predictor_gguf
                self.predictor_gguf = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                print("JoyCaption (GGUF): Old model unloaded (model switch - custom node).")

            try:
                self.predictor_gguf = JoyCaptionPredictorGGUF(model_name=gguf_model, mmproj_name=mmproj_file, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
                self.current_model_key = model_key
            except Exception as e:
                print(f"JoyCaption (GGUF Custom): Node-level error loading GGUF model: {e}")
                return (f"Error loading GGUF model: {e}",)

        final_user_query = user_query.strip()
        
        if extra_options_input:
            if isinstance(extra_options_input, tuple) and len(extra_options_input) == 2:
                extras, person_name_from_options = extra_options_input
                if not isinstance(extras, list): extras = []
                if not isinstance(person_name_from_options, str): person_name_from_options = ""

                processed_extra_options = []
                for opt_str in extras:
                    try:
                        processed_extra_options.append(opt_str.format(name=person_name_from_options or "{NAME}"))
                    except KeyError as e_opt:
                        if 'name' not in str(e_opt).lower(): # Check if the error is specifically about 'name'
                            print(f"JoyCaption (GGUF Custom) Warning: Extra option formatting error: '{opt_str}'. Missing key: {e_opt}")
                            processed_extra_options.append(opt_str + f" (Extra option formatting error: missing key {e_opt})")
                        else: # If it's just {name} missing and name_input is empty, it's fine.
                            processed_extra_options.append(opt_str)
                
                if processed_extra_options:
                    final_user_query += " " + " ".join(processed_extra_options)
            else:
                print(f"JoyCaption (GGUF Custom) Warning: extra_options_input is not in the expected format (list, str). Received: {type(extra_options_input)}")
        
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        response = self.predictor_gguf.generate(pil_image, system_prompt.strip(), final_user_query, max_new_tokens, temperature, top_p, top_k)

        if unload_after_generate and self.predictor_gguf is not None:
            if hasattr(self.predictor_gguf, 'chat_handler_exit_stack') and self.predictor_gguf.chat_handler_exit_stack is not None:
                try:
                    print("JoyCaption (GGUF): Manually closing chat_handler_exit_stack (unload_after_generate - custom node).")
                    self.predictor_gguf.chat_handler_exit_stack.close()
                except Exception as e_close:
                    print(f"JoyCaption (GGUF): Error closing chat_handler_exit_stack (unload_after_generate - custom node): {e_close}")
                self.predictor_gguf.chat_handler_exit_stack = None

            if hasattr(self.predictor_gguf, 'llm') and self.predictor_gguf.llm is not None:
                del self.predictor_gguf.llm
                self.predictor_gguf.llm = None # Explicitly set to None
            
            del self.predictor_gguf
            self.predictor_gguf = None
            self.current_model_key = None # Crucial to reset this
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("JoyCaption (GGUF): Model unloaded, chat_handler_exit_stack closed, GC run, CUDA cache emptied (unload_after_generate - custom node).")
            
        return (response,)
