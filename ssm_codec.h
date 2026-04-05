#ifndef SSM_CODEC_H
#define SSM_CODEC_H

int do_compress(const char *inpath, const char *outpath);
int do_decompress(const char *inpath, const char *outpath);
int do_verify(const char *inpath);

#endif /* SSM_CODEC_H */
