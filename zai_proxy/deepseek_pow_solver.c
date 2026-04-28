#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RATE_BYTES 136
#define MASK64 UINT64_MAX

static const unsigned int ROT[5][5] = {
    {0, 36, 3, 41, 18},
    {1, 44, 10, 45, 2},
    {62, 6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39, 8, 14},
};

static const uint64_t ROUND_CONSTANTS[24] = {
    0x0000000000000001ULL,
    0x0000000000008082ULL,
    0x800000000000808AULL,
    0x8000000080008000ULL,
    0x000000000000808BULL,
    0x0000000080000001ULL,
    0x8000000080008081ULL,
    0x8000000000008009ULL,
    0x000000000000008AULL,
    0x0000000000000088ULL,
    0x0000000080008009ULL,
    0x000000008000000AULL,
    0x000000008000808BULL,
    0x800000000000008BULL,
    0x8000000000008089ULL,
    0x8000000000008003ULL,
    0x8000000000008002ULL,
    0x8000000000000080ULL,
    0x000000000000800AULL,
    0x800000008000000AULL,
    0x8000000080008081ULL,
    0x8000000000008080ULL,
    0x0000000080000001ULL,
    0x8000000080008008ULL,
};

static uint64_t rol64(uint64_t value, unsigned int offset) {
    offset %= 64;
    if (offset == 0) {
        return value;
    }
    return (value << offset) | (value >> (64 - offset));
}

static uint64_t load_le64(const unsigned char *data) {
    uint64_t value = 0;
    for (unsigned int i = 0; i < 8; i++) {
        value |= ((uint64_t)data[i]) << (8 * i);
    }
    return value;
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

static int parse_expected_lanes(const char *hex, uint64_t expected[4]) {
    if (strlen(hex) != 64) {
        return 0;
    }

    unsigned char bytes[32];
    for (unsigned int i = 0; i < 32; i++) {
        int high = hex_value(hex[2 * i]);
        int low = hex_value(hex[2 * i + 1]);
        if (high < 0 || low < 0) {
            return 0;
        }
        bytes[i] = (unsigned char)((high << 4) | low);
    }

    for (unsigned int lane = 0; lane < 4; lane++) {
        expected[lane] = load_le64(bytes + lane * 8);
    }
    return 1;
}

static void keccak_f_deepseek(uint64_t state[25]) {
    for (unsigned int round = 1; round < 24; round++) {
        uint64_t columns[5];
        uint64_t deltas[5];
        uint64_t rotated[25];

        for (unsigned int x = 0; x < 5; x++) {
            columns[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (unsigned int x = 0; x < 5; x++) {
            deltas[x] = columns[(x + 4) % 5] ^ rol64(columns[(x + 1) % 5], 1);
        }
        for (unsigned int x = 0; x < 5; x++) {
            for (unsigned int y = 0; y < 5; y++) {
                state[x + 5 * y] ^= deltas[x];
            }
        }

        memset(rotated, 0, sizeof(rotated));
        for (unsigned int x = 0; x < 5; x++) {
            for (unsigned int y = 0; y < 5; y++) {
                rotated[y + 5 * ((2 * x + 3 * y) % 5)] = rol64(state[x + 5 * y], ROT[x][y]);
            }
        }

        for (unsigned int x = 0; x < 5; x++) {
            for (unsigned int y = 0; y < 5; y++) {
                state[x + 5 * y] =
                    rotated[x + 5 * y] ^
                    ((~rotated[((x + 1) % 5) + 5 * y]) & rotated[((x + 2) % 5) + 5 * y]);
            }
        }

        state[0] ^= ROUND_CONSTANTS[round];
    }
}

static int hash_matches(
    const char *salt,
    unsigned long long expire_at,
    unsigned int answer,
    const uint64_t expected[4]
) {
    unsigned char block[RATE_BYTES];
    uint64_t state[25];
    memset(block, 0, sizeof(block));
    memset(state, 0, sizeof(state));

    int len = snprintf((char *)block, sizeof(block), "%s_%llu_%u", salt, expire_at, answer);
    if (len <= 0 || len >= RATE_BYTES) {
        return 0;
    }

    block[len] ^= 0x06;
    block[RATE_BYTES - 1] ^= 0x80;

    for (unsigned int lane = 0; lane < RATE_BYTES / 8; lane++) {
        state[lane] ^= load_le64(block + lane * 8);
    }

    keccak_f_deepseek(state);
    return state[0] == expected[0] &&
           state[1] == expected[1] &&
           state[2] == expected[2] &&
           state[3] == expected[3];
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s CHALLENGE_HEX SALT EXPIRE_AT DIFFICULTY\n", argv[0]);
        return 64;
    }

    uint64_t expected[4];
    if (!parse_expected_lanes(argv[1], expected)) {
        fprintf(stderr, "invalid challenge hex\n");
        return 65;
    }

    char *end = NULL;
    unsigned long long expire_at = strtoull(argv[3], &end, 10);
    if (!end || *end != '\0') {
        fprintf(stderr, "invalid expire_at\n");
        return 66;
    }

    end = NULL;
    unsigned long difficulty = strtoul(argv[4], &end, 10);
    if (!end || *end != '\0' || difficulty == 0 || difficulty > UINT32_MAX) {
        fprintf(stderr, "invalid difficulty\n");
        return 67;
    }

    for (unsigned int answer = 0; answer < (unsigned int)difficulty; answer++) {
        if (hash_matches(argv[2], expire_at, answer, expected)) {
            printf("%u\n", answer);
            return 0;
        }
    }

    fprintf(stderr, "no solution found\n");
    return 2;
}
