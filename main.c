#include "ssm_config.h"
#include "ssm_tokenizer.h"
#include "ssm_codec.h"
#include "ssm_preprocess.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [--pp=none|reverse|bwt|extbpe] c|compress <in> <out>\n"
                        "       %s [--pp=...] d|decompress <in.ssm> <out>\n"
                        "       %s [--pp=...] v|verify <in>\n", argv[0], argv[0], argv[0]);
        return 1;
    }

    /* Parse preprocessing flag */
    int arg_offset = 1;
    g_pp_mode = PP_NONE;
    if (argc > 1 && strncmp(argv[1], "--pp=", 5) == 0) {
        const char *mode = argv[1] + 5;
        if (!strcmp(mode, "none"))    g_pp_mode = PP_NONE;
        else if (!strcmp(mode, "reverse")) g_pp_mode = PP_REVERSE;
        else if (!strcmp(mode, "bwt"))     g_pp_mode = PP_BWT;
        else if (!strcmp(mode, "extbpe"))  g_pp_mode = PP_EXTBPE;
        else { fprintf(stderr, "Unknown pp mode: %s\n", mode); return 1; }
        arg_offset = 2;
    }

    if (argc < arg_offset + 1) {
        fprintf(stderr, "Missing command\n");
        return 1;
    }

    /* Load tokenizer */
    const char *tok_path = "tokenizer/tokenizer.bin";
    printf("Loading tokenizer from %s ...\n", tok_path);
    if (load_tokenizer(tok_path)) return 1;
    printf("Tokenizer loaded (vocab=%d)\n", VOCAB);
    if (g_pp_mode != PP_NONE) {
        const char *names[] = {"none","reverse","bwt","extbpe"};
        printf("Preprocessing: %s\n", names[g_pp_mode]);
    }

    if (!strcmp(argv[arg_offset], "c") || !strcmp(argv[arg_offset], "compress")) {
        if (argc != arg_offset + 3) { fprintf(stderr, "Usage: %s [--pp=...] c <in> <out>\n", argv[0]); return 1; }
        return do_compress(argv[arg_offset+1], argv[arg_offset+2]);
    }
    if (!strcmp(argv[arg_offset], "d") || !strcmp(argv[arg_offset], "decompress")) {
        if (argc != arg_offset + 3) { fprintf(stderr, "Usage: %s [--pp=...] d <in.ssm> <out>\n", argv[0]); return 1; }
        return do_decompress(argv[arg_offset+1], argv[arg_offset+2]);
    }
    if (!strcmp(argv[arg_offset], "v") || !strcmp(argv[arg_offset], "verify")) {
        if (argc != arg_offset + 2) { fprintf(stderr, "Usage: %s [--pp=...] v <in>\n", argv[0]); return 1; }
        return do_verify(argv[arg_offset+1]);
    }
    fprintf(stderr, "Unknown command: %s\n", argv[arg_offset]);
    return 1;
}
