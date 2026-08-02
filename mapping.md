# Hafs and Qalun Mapping Architecture

The Quran was revealed in seven ahruf and transmitted in various Qira'at. Two of the most commonly recited transmissions today are **Hafs 'an 'Asim** and **Qalun 'an Nafi'**. 

This application supports both, allowing the user to select their preferred Riwayah. Since automatic speech recognition (ASR) engines typically map spoken Arabic back to a standard modern or classical lexicon (which aligns closely with the Hafs orthography), the app requires a specialized mapping mechanism to correctly evaluate a Qalun recitation.

## The Challenge

1. **Orthographic Differences:** Words are sometimes spelled or pronounced differently (e.g., adding an Alif, reading a letter with a different vowel, or adding a plural Silah like "همو" instead of "هم").
2. **ASR Limitations:** ASR might transcribe the Qalun recitation (e.g., "بِيُوتِ" or "مَلِكِ") into the standard recognizable text it was trained on, which is usually Hafs (e.g., "بُيُوتِ" or "مَالِكِ").
3. **Word Alignment:** We need to evaluate what the user *actually* said against what they were *supposed* to say in their chosen Riwayah, without failing them if the ASR standardizes their speech.

## How Mapping Works

The application employs a robust text normalization and fuzzy matching algorithm to bridge the gap between spoken words, the intended Riwayah text, and the ASR transcription.

1. **Normalization:** 
   Both the target text (from the API) and the spoken text (from the ASR) undergo heavy normalization before comparison.
   - Diacritics (Tashkeel) are stripped.
   - Tatweel (stretch characters) are removed.
   - Variations of Alif (أ, إ, آ) are normalized to bare Alif (ا).
   - Variations of Ya/Alif Maqsura (ي, ى), Ta Marbutah/Ha (ة, ه), and Waw/Hamza are standardized.

2. **Expected Phonetic Substitutions (Qalun vs Hafs Index):**
   If Qalun is selected, the matching algorithm expects specific phonetic footprints:
   - **Silah Mim Al-Jam الجمع ميم صلة:** The ASR might capture "همو" or "كمو", which would fail a strict standard match against "هم" or "كم".
   - **Specific Word Adjustments:** Words like "يُؤَدِّهِ" vs "يُؤَدِّهِي" or "بُيُوتِ" vs "بِيُوتِ".
   - **Hamza Rules:** Facilitation (Tashil) or omission of Hamza. 
   
3. **Fuzzy String Alignment:**
   Instead of demanding an exact character match, the engine iterates through the spoken phrase using a lookahead mechanism (to account for skipped words, stutters, or split/merged ASR outputs). When matching:
   - It computes the Levenshtein distance or evaluates substring structures between the normalized user speech and the normalized target word.
   - A tolerance threshold ensures that if the ASR output is very close (e.g., missed one letter that is known to be structurally ambiguous in the chosen Riwayah), it gives the user credit for reading the correct rule.
   - If the user skips a word, it maintains its pointer and continues matching the next target words, securely marking the missed word as `SKIPPED`.

The result is a forgiving, Riwayah-aware alignment that recognizes the structural intent of the recitation rather than rigidly comparing raw API text to raw ASR output.
