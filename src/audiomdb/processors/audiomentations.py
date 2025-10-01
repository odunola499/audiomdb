from audiomdb.processors import AudioProcessor
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

test_augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5),
])


class AudiomentationsProcessor(AudioProcessor):
    """
    Audio processor that applies augmentations using the audiomentations library.
    Example:
        from audiomentations import Compose, AddGaussianNoise, TimeStretch
        augmentations = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)
        ])
        processor = AudiomentationsProcessor(augmentations=augmentations, keep_original=True)
        processed_audio = processor.process(audio_data)
    """
    def __init__(self, augmentations, keep_original: bool = False):
        super().__init__(keep_original)
        self.augmentations = augmentations

    def process(self, data, sample_rate = 16000):
        """
        Apply the defined augmentations to the input audio data.
        Args:
            data: Input audio data to be processed.
        Returns:
            Augmented audio data.
            :param data:
            :param sample_rate:
        """
        augmented_data = self.augmentations(samples=data, sample_rate=sample_rate)
        return {
            'audio': augmented_data
        }