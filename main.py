import gradio as gr
import whisper


def trans_2x(audio_file, model_type):
    model = whisper.load_model(model_type[0])
    result = model.transcribe(audio_file)
    translation = model.transcribe(audio_file, task="translate")
    return result["text"], translation["text"]


def main():
    audio_input = [
        gr.inputs.Audio(source="upload", type="filepath", label="Audio File"),
        gr.CheckboxGroup(
            ["tiny", "small", "base", "medium", "large"],
            label="Models",
            info="Choose model quality, only the smallest selected model will be used.",
        ),
    ]

    output_text = [
        gr.outputs.Textbox(
            label="Transcript"
        ),
        gr.outputs.Textbox(
            label="Translation"
        ),
    ]

    iface = gr.Interface(
        fn=trans_2x, inputs=audio_input, outputs=output_text, title="Whisper UI"
    )

    iface.launch()


if __name__ == "__main__":
    main()
