from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import pronouncing 

def transcribe_to_phonemes(text):
    words = text.split(" ")
    stripped_phonemes = []
    for word in words:
        pronounciations = pronouncing.phones_for_word(word)
        phonemes = pronounciations[0].split(" ")
        vowels = ['A', 'E', 'I', 'O', 'U']
        for phoneme in phonemes:
            if phoneme[0] in vowels:
                phoneme = phoneme[:-1]
                print(phoneme)
            stripped_phonemes.append(phoneme)
    # Flatten the list and filter out empty phoneme lists
    return stripped_phonemes

text = "hello world"
phonemes = transcribe_to_phonemes(text)
video = []

target_resolution = (1280, 720)  

clip1 = VideoFileClip("phonemes/" + "AA" + ".mp4")
print(type(clip1))

clip2 = VideoFileClip("phonemes/" + "AE" + ".mp4").resize(newsize=target_resolution)
clip3 = VideoFileClip("phonemes/" + "AH" + ".mp4").resize(newsize=target_resolution)
clip4 = VideoFileClip("phonemes/" + "AO" + ".mp4").resize(newsize=target_resolution)

final_clip = concatenate_videoclips([clip1, clip2, clip3, clip4])

final_clip

# Write the result to a file
final_clip.write_videofile("phonemes/output_video1.mp4", codec = "libx264")

