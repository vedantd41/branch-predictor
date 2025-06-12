#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "predictor.h"

const char *bpName[4] = { "Static", "Gshare", "Tournament", "Custom" };

int ghistoryBits;
int lhistoryBits;
int pcIndexBits;
int bpType;
int verbose;

//------------------ GShare ------------------//
uint32_t ghr = 0;
uint8_t *gshareTable;

//------------------ Tournament -------------//
uint32_t *localHistoryTable;
uint8_t *localPredictorTable;
uint8_t *globalPredictorTable;
uint8_t *chooserTable;
uint32_t globalHistory = 0;

//------------------ Custom (Perceptron) ------------------//
#define NUM_PERCEPTRONS 256 //256
#define HISTORY_LEN     28
#define THRESHOLD       50 //21

int8_t perceptronTable[NUM_PERCEPTRONS][HISTORY_LEN + 1]; // +1 for bias
uint32_t perceptronHistory = 0;

void init_predictor() {
  uint32_t gsize = 1 << ghistoryBits;
  gshareTable = (uint8_t *)malloc(gsize * sizeof(uint8_t));
  for (int i = 0; i < gsize; i++) gshareTable[i] = WN;

  if (bpType == CUSTOM || bpType == TOURNAMENT) {
    uint32_t lhistSize = 1 << pcIndexBits;
    uint32_t lpredSize = 1 << lhistoryBits;
    uint32_t gpredSize = 1 << ghistoryBits;

    localHistoryTable = (uint32_t *)malloc(lhistSize * sizeof(uint32_t));
    localPredictorTable = (uint8_t *)malloc(lpredSize * sizeof(uint8_t));
    globalPredictorTable = (uint8_t *)malloc(gpredSize * sizeof(uint8_t));
    chooserTable = (uint8_t *)malloc(gpredSize * sizeof(uint8_t));

    for (int i = 0; i < lhistSize; i++) localHistoryTable[i] = 0;
    for (int i = 0; i < lpredSize; i++) localPredictorTable[i] = WN;
    for (int i = 0; i < gpredSize; i++) globalPredictorTable[i] = WN;
    for (int i = 0; i < gpredSize; i++) chooserTable[i] = WN;
  }

  if (bpType == CUSTOM) {
    memset(perceptronTable, 0, sizeof(perceptronTable));
    perceptronHistory = 0;
  }
}

uint8_t make_prediction(uint32_t pc) {
  switch (bpType) {
    case STATIC:
      return TAKEN;

    case GSHARE: {
      uint32_t index = (pc ^ ghr) & ((1 << ghistoryBits) - 1);
      return gshareTable[index] >= WT ? TAKEN : NOTTAKEN;
    }

    case TOURNAMENT: {
      uint32_t pc_index = pc & ((1 << pcIndexBits) - 1);
      uint32_t lhist = localHistoryTable[pc_index];
      uint8_t lpred = localPredictorTable[lhist];
      uint32_t gindex = globalHistory & ((1 << ghistoryBits) - 1);
      uint8_t gpred = globalPredictorTable[gindex];
      uint8_t choice = chooserTable[gindex];

      return (choice >= WT) ? (gpred >= WT ? TAKEN : NOTTAKEN)
                            : (lpred >= WT ? TAKEN : NOTTAKEN);
    }

    case CUSTOM: {
      uint32_t index = pc % NUM_PERCEPTRONS;
      int32_t sum = perceptronTable[index][0]; // bias

      for (int i = 0; i < HISTORY_LEN; i++) {
        int bit = (perceptronHistory >> i) & 1;
        sum += perceptronTable[index][i + 1] * (bit ? 1 : -1);
      }

      return sum >= 0 ? TAKEN : NOTTAKEN;
    }

    default:
      return NOTTAKEN;
  }
}

void train_predictor(uint32_t pc, uint8_t outcome) {
  switch (bpType) {
    case GSHARE: {
      uint32_t index = (pc ^ ghr) & ((1 << ghistoryBits) - 1);
      if (outcome == TAKEN && gshareTable[index] < ST) gshareTable[index]++;
      if (outcome == NOTTAKEN && gshareTable[index] > SN) gshareTable[index]--;
      ghr = ((ghr << 1) | outcome) & ((1 << ghistoryBits) - 1);
      break;
    }

    case TOURNAMENT: {
      uint32_t pc_index = pc & ((1 << pcIndexBits) - 1);
      uint32_t lhist = localHistoryTable[pc_index];
      uint8_t lpred = localPredictorTable[lhist];
      uint32_t gindex = globalHistory & ((1 << ghistoryBits) - 1);
      uint8_t gpred = globalPredictorTable[gindex];
      uint8_t choice = chooserTable[gindex];

      uint8_t localDecision = lpred >= WT ? TAKEN : NOTTAKEN;
      uint8_t globalDecision = gpred >= WT ? TAKEN : NOTTAKEN;

      if (outcome == TAKEN && localPredictorTable[lhist] < ST) localPredictorTable[lhist]++;
      if (outcome == NOTTAKEN && localPredictorTable[lhist] > SN) localPredictorTable[lhist]--;

      if (outcome == TAKEN && globalPredictorTable[gindex] < ST) globalPredictorTable[gindex]++;
      if (outcome == NOTTAKEN && globalPredictorTable[gindex] > SN) globalPredictorTable[gindex]--;

      if (localDecision != globalDecision) {
        if (globalDecision == outcome && chooserTable[gindex] < ST)
          chooserTable[gindex]++;
        else if (localDecision == outcome && chooserTable[gindex] > SN)
          chooserTable[gindex]--;
      }

      localHistoryTable[pc_index] = ((lhist << 1) | outcome) & ((1 << lhistoryBits) - 1);
      globalHistory = ((globalHistory << 1) | outcome) & ((1 << ghistoryBits) - 1);
      break;
    }

    case CUSTOM: {
      uint32_t index = pc % NUM_PERCEPTRONS;
      int32_t sum = perceptronTable[index][0];

      for (int i = 0; i < HISTORY_LEN; i++) {
        int bit = (perceptronHistory >> i) & 1;
        sum += perceptronTable[index][i + 1] * (bit ? 1 : -1);
      }

      uint8_t prediction = (sum >= 0) ? TAKEN : NOTTAKEN;
      int target = (outcome == TAKEN) ? 1 : -1;

      // Train only if prediction wrong or low confidence
      if ((prediction != outcome) || (abs(sum) < THRESHOLD)) {
        if (perceptronTable[index][0] + target <= 127 && perceptronTable[index][0] + target >= -128)
          perceptronTable[index][0] += target;

        for (int i = 0; i < HISTORY_LEN; i++) {
          int bit = (perceptronHistory >> i) & 1;
          int direction = bit ? 1 : -1;
          if (perceptronTable[index][i + 1] + target * direction <= 127 &&
              perceptronTable[index][i + 1] + target * direction >= -128) {
            perceptronTable[index][i + 1] += target * direction;
          }
        }
      }

      perceptronHistory = ((perceptronHistory << 1) | (outcome == TAKEN)) & ((1ULL << HISTORY_LEN) - 1);
      break;
    }

    default:
      break;
  }
}
