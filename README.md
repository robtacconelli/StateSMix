# StateSMix

**Online Lossless Compression via Mamba State Space Models and Sparse N-gram Context Mixing**

StateSMix is a fully self-contained lossless compressor combining an online-trained Mamba SSM with sparse n-gram logit biasing (bigram through 32-gram) and arithmetic coding. No pre-trained weights, no GPU, and no external dependencies are required.

## Results on enwik8

| File | StateSMix | xz -9e | Delta |
|------|-----------|--------|-------|
| 1 MB | 265,370 B (2.123 bpb) | 2.326 bpb | **-8.7%** |
| 3 MB | 805,926 B (2.149 bpb) | 2.271 bpb | **-5.4%** |
| 10 MB | 2,702,498 B (2.162 bpb) | 2.177 bpb | **-0.7%** |
| 100 MB | 26,622,640 B (2.130 bpb) | 1.992 bpb | +6.9% |

StateSMix beats xz on all file sizes up to 10 MB.

## Building

```bash
make
```

Requires GCC with AVX2/FMA support. OpenMP is used for parallel training.

## Usage

```bash
# Compress
./ssm_best_version2 c input_file output_file.ssm

# Decompress
./ssm_best_version2 d output_file.ssm recovered_file

# Verify (compress + decompress + compare)
./ssm_best_version2 v input_file
```

## Architecture

- **SSM**: Mamba-style (DM=32, DS=16, DI=64, NL=2), ~120K parameters, trained online with Adam
- **N-gram tables**: Bigram through 32-gram with softmax-invariant sparse logit bias
- **Arithmetic coding**: 32-bit range coder, AC_SCALE=2^16
- **Tokenization**: GPT-NeoX BPE (49,152 types) with compact vocabulary remapping

See `architecture.txt` for detailed documentation and `ssm_compress_paper.tex` for the research paper.

## Requirements

- GCC with `-mavx2 -mfma` support
- ~6 GB RAM for 100 MB input files
- Tokenizer binary in `tokenizer/tokenizer.bin`

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Author

Roberto Tacconelli (tacconelli.rob@gmail.com)
