"""
Create synthetic MIDI dataset:
- Two parts per file: melody (monophonic quarter notes) and harmony (3-voice diatonic triads).
- Key: C major, Time signature: 4/4
- 8 bars per piece, 4 quarter notes per bar -> 32 quarter notes per piece
"""

import os
import random
from music21 import stream, note, chord, metadata, key as mkey, meter, tempo, instrument, pitch

def build_c_major_scale_midis(octave_min=3, octave_max=6):
    """
    Build a sorted list of MIDI numbers corresponding to C major notes
    across octaves [octave_min..octave_max] (inclusive).
    Order is increasing pitch.
    """
    scale_names = ['C','D','E','F','G','A','B']
    midis = []
    for octv in range(octave_min, octave_max + 1):
        for nm in scale_names:
            p = pitch.Pitch(f"{nm}{octv}")
            midis.append(p.midi)
    midis = sorted(midis)
    return midis

def make_diatonic_triad_from_scale_list(scale_midis, root_midi_index):
    """
    Given the scale_midis list and index of the root within that list,
    return the 3-voice triad as a list of midi numbers: [root, third, fifth].
    We pick indices [i, i+2, i+4] to form a diatonic triad.
    Assumes the scale_midis spans enough octaves so that i+4 exists.
    """
    i = root_midi_index
    # Safety: if near end, wrap by subtracting 7 (one scale octave) until fits
    L = len(scale_midis)
    if i + 4 >= L:
        # Move index down until i+4 < L
        # This keeps the root where it is but ensures we can get triad above it.
        # Ideally scale list spans enough octaves for this not to be needed.
        while i + 4 >= L and i - 7 >= 0:
            i = i - 7
    # Now pick i, i+2, i+4
    triad = [scale_midis[i], scale_midis[i+2], scale_midis[i+4]]
    return triad

def create_synthetic_dataset(dest_folder, n_pieces=1000, seed=42,
                             melody_octaves=(4,5),           # where melody notes come from
                             scale_octaves=(3,7),            # scale octave span for chord construction
                             bars=8, beats_per_bar=4,
                             tempo_bpm=100):
    """
    Create n_pieces MIDI files under dest_folder.
    """
    random.seed(seed)

    os.makedirs(dest_folder, exist_ok=True)
    total_quarters = bars * beats_per_bar

    # Build scale midis spanning octaves for chord building
    scale_midis = build_c_major_scale_midis(octave_min=scale_octaves[0], octave_max=scale_octaves[1])

    # Build a melody candidate list: C major notes in the melody_octaves range
    melody_candidates = []
    for octv in range(melody_octaves[0], melody_octaves[1] + 1):
        for nm in ['C','D','E','F','G','A','B']:
            melody_candidates.append(pitch.Pitch(f"{nm}{octv}").midi)
    melody_candidates = sorted(melody_candidates)

    for idx in range(n_pieces):
        # --- Score and meta ---
        sc = stream.Score()
        sc.insert(0, metadata.Metadata())
        sc.metadata.title = f"Synthetic piece {idx:04d}"
        sc.append(tempo.MetronomeMark(number=tempo_bpm))

        # --- Melody part ---
        mel_part = stream.Part()
        mel_part.append(instrument.Piano())  # optional, helps some DAWs
        mel_part.append(meter.TimeSignature(f"{beats_per_bar}/4"))
        mel_part.append(mkey.Key('C'))

        # --- Harmony part ---
        harm_part = stream.Part()
        harm_part.append(instrument.Piano())
        harm_part.append(meter.TimeSignature(f"{beats_per_bar}/4"))
        harm_part.append(mkey.Key('C'))

        # Fill notes: quarter length = 1.0
        offset = 0.0
        for t in range(total_quarters):
            # pick melody note randomly from melody_candidates
            m_midi = random.choice(melody_candidates)
            m_note = note.Note(m_midi)
            m_note.quarterLength = 1.0
            # Optionally set octave/velocity details if desired
            mel_part.insert(offset, m_note)

            # Build chord whose root is this melody note (diatonic triad)
            # find index of m_midi in scale_midis (we have many octave reps)
            # choose the *first* occurrence from low to high that matches m_midi and that leaves room for i+4
            possible_indices = [i for i, mm in enumerate(scale_midis) if mm == m_midi]
            if len(possible_indices) == 0:
                # Fallback: find nearest scale degree (shouldn't happen as melody chosen from scale)
                # Choose the closest scale midi
                closest_idx = min(range(len(scale_midis)), key=lambda i: abs(scale_midis[i] - m_midi))
                triad_midis = make_diatonic_triad_from_scale_list(scale_midis, closest_idx)
            else:
                # pick the occurrence with sufficient room
                chosen_index = None
                for candidate_idx in possible_indices:
                    if candidate_idx + 4 < len(scale_midis):
                        chosen_index = candidate_idx
                        break
                if chosen_index is None:
                    # fallback to first occurrence and allow the helper to adjust
                    chosen_index = possible_indices[0]
                triad_midis = make_diatonic_triad_from_scale_list(scale_midis, chosen_index)

            # Convert to music21 chord
            chord_pitches = [pitch.Pitch(m) for m in triad_midis]
            c = chord.Chord(chord_pitches)
            c.quarterLength = 1.0
            # Optionally set voicing: move bass down an octave to give nicer spacing
            # e.g., lower the root by 12 semitones if it is too high
            # Here we leave as-is so root equals melody pitch
            harm_part.insert(offset, c)

            offset += 1.0

        # Insert parts into score
        sc.insert(0, mel_part)
        sc.insert(0, harm_part)

        # write midi
        fname = os.path.join(dest_folder, f"synthetic_piece_{idx:04d}.mid")
        try:
            sc.write('midi', fp=fname)
        except Exception as e:
            # music21 can sometimes fail on weird platform configs; try an alternative fallback
            print(f"Warning: failed to write MIDI for {fname}: {e}")

        if (idx + 1) % 50 == 0 or idx == n_pieces - 1:
            print(f"Created {idx + 1}/{n_pieces}: {fname}")

    print("Done. Created", n_pieces, "files in", dest_folder)


if __name__ == "__main__":
    # Example usage:
    # out_dir = "/media/maindisk/data/synthetic_dataset_midi"
    out_dir = "/media/maindisk/data/synthetic_CA_test"
    create_synthetic_dataset(out_dir, n_pieces=100, seed=123,
                             melody_octaves=(4,5), scale_octaves=(3,8),
                             bars=8, beats_per_bar=4, tempo_bpm=100)
