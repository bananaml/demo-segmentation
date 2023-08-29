![](https://www.banana.dev/lib_zOkYpJoyYVcAamDf/x2p804nk9qvjb1vg.svg?w=340 "Banana.dev")

# Banana.dev bark starter template

This is a segmentation starter template from [Banana.dev](https://www.banana.dev) that allows on-demand serverless GPU inference.

You can fork this repository and deploy it on Banana as is, or customize it based on your own needs.


# Running this app

## Deploying on Banana.dev

1. [Fork this](https://github.com/bananaml/demo-segmentation/fork) repository to your own GitHub account.
2. Connect your GitHub account on Banana.
3. [Create a new model](https://app.banana.dev/deploy) on Banana from the forked GitHub repository.

## Running after deploying

1. Wait for the model to build after creating it.
2. Make an API request using one of the provided snippets in your Banana dashboard. However, instead of sending a prompt as provided in the snippet, adjust the prompt to fit the needs of the segmentation model:

```python
inputs = {
    "audio": "bucket_link_to_wav_file",
    "option": "voice_activity_detection"
}
```

The `audio` parameter should be substituted with your S3 (or any other provider where you can store .wav files) bucket link that contains the .wav audio file you want to segment. For the `option` parameter, you have to choose between the following options depending on what segmentation information you want to gain from the audio file:

* voice_activity_detection
* overlapped_speech_detection
* instantaneous_speaker_counting
* speaker_change_detection

In the example above, we chose `voice_activity_detection` as an option.

For more info, check out the [Banana.dev docs](https://docs.banana.dev/banana-docs/).