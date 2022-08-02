import glob
import json

from tokenizers import (Tokenizer, decoders, models, normalizers,
                        pre_tokenizers, processors, trainers)


def common_crawl(dir):
    # commoncrawl = "/mnt/cfs/commoncrawl-202*-**-s3-filter/"
    # dirs = ["/mnt/cfs/commoncrawl-2021-03-filter/minhash"]
    for file in glob.glob(dir + "/**"):
        f = open(file, "r")
        items = f.readlines()
        for jsonl in items:
            doc = json.loads(jsonl)
            for cont in doc["cont"]:
                yield cont


def weibo():
    # weibo
    weibo = "/mnt/cfs/weibo_comments/processed"
    for file in glob.glob(weibo + "/**"):
        f = open(file, "r")
        items = json.load(f)[0]
        for conv in items:
            # conv = [" ".join([w for w in c]) for c in conv["texts"]]
            # plain_conv = " <|endoftext|> ".join(conv)
            # yield plain_conv
            yield "\n".join(conv['texts'])

def lccc():
    # lccc
    lccc = "/mnt/cfs/LCCC"
    for file in glob.glob(lccc + "/**"):
        f = open(file, "r")
        items = json.load(f)
        for conv in items:
            yield "\n".join(conv)


def main():
    # # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # # Customize training
    # pre_tokenizer = pre_tokenizers.Sequence(
    # [pre_tokenizers.CharDelimiterSplit(" ")]
    # )
    # # pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=[
                                  "<pad>", "<|endoftext|>", "<EMAIL>", "<PHONE>"])

    weibo_iterator = weibo()
    lccc_iterator = lccc()
    common_crawl_iterator = [common_crawl("/mnt/cfs/commoncrawl-2021-12-s3-filter/")]
    tokenizer.train_from_iterator(weibo_iterator, trainer=trainer)
    tokenizer.train_from_iterator(lccc_iterator, trainer=trainer)
    for iter in common_crawl_iterator:
        tokenizer.train_from_iterator(iter, trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.save("./gpt_bpe.json")


if __name__ == "__main__":
    main()
