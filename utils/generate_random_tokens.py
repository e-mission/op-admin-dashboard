import secrets
import argparse


def generateRandomToken(length, outFormat):
    if outFormat == "hex":
        return secrets.token_hex(length)
    else:
        return secrets.token_urlsafe(length)

def generateRandomTokensForProgram(program, token_length, count, outFormat):
    return ["%s_%s" % (program,generateRandomToken(token_length, outFormat)) for i in range(count)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="generate_random_tokens")

    parser.add_argument("program")
    parser.add_argument("token_length", type=int)
    parser.add_argument("count", type=int)
    parser.add_argument("outFormat", choices=["urlsafe", "hex", "base64"])

    args = parser.parse_args()

    tokens = generateRandomTokensForProgram(args.program, args.token_length, args.count, args.outFormat)
    for t in tokens:
        print(t)
