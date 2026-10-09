#pragma once

#include "kernel_operator.h"

namespace SglLim {
using namespace AscendC;

class ResidentManager {
public:
    __aicore__ inline void Init(GM_ADDR ids, GM_ADDR scores, GM_ADDR ends, GM_ADDR lengths,
        GM_ADDR table, GM_ADDR requests, GM_ADDR generations, GM_ADDR mapping, GM_ADDR sources,
        GM_ADDR physical, GM_ADDR state, GM_ADDR outputIds, GM_ADDR outputSlots, GM_ADDR membership,
        GM_ADDR unionSources, GM_ADDR unionSlots, GM_ADDR unionPhysical, uint32_t rows, uint32_t batch,
        uint32_t requestCapacity, uint32_t sourceCapacity, uint32_t residentCapacity, uint32_t columns,
        uint32_t unionWidth, uint32_t scoreColumns, uint64_t hostTokens, uint32_t mutableTail,
        bool pooledInput, bool encodedScores, TPipe &pipe, bool preparedRows = false)
    {
        idsGm.SetGlobalBuffer((__gm__ int32_t *)ids);
        scoresGm.SetGlobalBuffer((__gm__ uint32_t *)scores);
        endsGm.SetGlobalBuffer((__gm__ int32_t *)ends);
        lengthsGm.SetGlobalBuffer((__gm__ int32_t *)lengths);
        tableGm.SetGlobalBuffer((__gm__ int32_t *)table);
        requestsGm.SetGlobalBuffer((__gm__ int64_t *)requests);
        generationsGm.SetGlobalBuffer((__gm__ int64_t *)generations);
        mappingGm.SetGlobalBuffer((__gm__ int32_t *)mapping);
        sourcesGm.SetGlobalBuffer((__gm__ int32_t *)sources);
        physicalGm.SetGlobalBuffer((__gm__ int64_t *)physical);
        stateGm.SetGlobalBuffer((__gm__ int64_t *)state);
        outputIdsGm.SetGlobalBuffer((__gm__ int32_t *)outputIds);
        outputSlotsGm.SetGlobalBuffer((__gm__ int32_t *)outputSlots);
        membershipGm.SetGlobalBuffer((__gm__ int16_t *)membership);
        unionSourcesGm.SetGlobalBuffer((__gm__ int32_t *)unionSources);
        unionSlotsGm.SetGlobalBuffer((__gm__ int32_t *)unionSlots);
        unionPhysicalGm.SetGlobalBuffer((__gm__ int64_t *)unionPhysical);
        rowCount = rows;
        batchCount = batch;
        requestCount = requestCapacity;
        sourceCount = sourceCapacity;
        capacity = residentCapacity;
        tableColumns = columns;
        unionSize = unionWidth;
        scoreStride = scoreColumns;
        hostSize = hostTokens;
        tailSize = mutableTail;
        pooled = pooledInput;
        encoded = encodedScores;
        prepared = preparedRows;
        pipe.InitBuffer(unionBuffer, unionSize * 4);
        pipe.InitBuffer(residentBuffer, capacity * 4);
        pipe.InitBuffer(heapBuffer, capacity * 4);
        pipe.InitBuffer(priorityBuffer, capacity * 4);
        pipe.InitBuffer(rowBuffer, 2080 * 16);
        unionLocal = unionBuffer.Get<int32_t>();
        residentLocal = residentBuffer.Get<int32_t>();
        heapLocal = heapBuffer.Get<int32_t>();
        priorityLocal = priorityBuffer.Get<float>();
        rowIds = rowBuffer.Get<int32_t>();
        rowSlots = rowIds[2080];
        rowPlan = rowIds[4160].ReinterpretCast<int16_t>();
    }

    __aicore__ inline void Run()
    {
        for (uint32_t request = GetBlockIdx(); request < batchCount; request += GetBlockNum() * 2) {
            ProcessRequest(request);
        }
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(mappingGm);
    }

private:
    __aicore__ inline void PublishUnion(uint32_t request, uint32_t unique, uint64_t mapBase)
    {
        int64_t owner = requestsGm.GetValue(request);
        if (owner > 0 && owner < requestCount) stateGm.SetValue(owner * 16 + 3, unique);
        LocalTensor<int64_t> rowPhysical = rowPlan.ReinterpretCast<int64_t>();
        for (uint32_t base = 0; base < unionSize; base += 1024) {
            Duplicate(rowIds, static_cast<int32_t>(-1), 1024);
            Duplicate(rowSlots, static_cast<int32_t>(-1), 1024);
            Duplicate(rowPlan.ReinterpretCast<int32_t>(), static_cast<int32_t>(-1), 2048);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            for (uint32_t local = 0; local < 1024 && base + local < unique; ++local) {
                int32_t token = unionLocal.GetValue(base + local);
                int32_t slot = mappingGm.GetValue(mapBase + token);
                int32_t page = tableGm.GetValue(static_cast<uint64_t>(request) * tableColumns + token / 128);
                rowIds.SetValue(local, token);
                rowSlots.SetValue(local, slot);
                rowPhysical.SetValue(local, static_cast<int64_t>(page) * 128 + token % 128);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID7);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID7);
            uint64_t offset = static_cast<uint64_t>(request) * unionSize + base;
            DataCopyPad(unionSourcesGm[offset], rowIds, {1, 4096, 0, 0});
            DataCopyPad(unionSlotsGm[offset], rowSlots, {1, 4096, 0, 0});
            DataCopyPad(unionPhysicalGm[offset], rowPhysical, {1, 8192, 0, 0});
            SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
        }
    }

    __aicore__ inline bool FinishPrepared(uint32_t request, int32_t start, int32_t end,
        int64_t owner, int32_t maxLength)
    {
        if (!prepared || owner <= 0 || owner >= requestCount) return false;
        uint32_t totalMisses = 0;
        for (int32_t row = start; row < end; ++row) {
            int16_t count = membershipGm.GetValue(static_cast<uint64_t>(row) * 16400 + 16391);
            if (count < 0) return false;
            totalMisses += count;
        }
        stateGm.SetValue(owner * 16 + 1, maxLength);
        stateGm.SetValue(owner * 16 + 2, 0);
        uint32_t unique = 0;
        if (totalMisses > 0) {
            Duplicate(residentLocal, static_cast<int32_t>(0), capacity);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            for (int32_t row = start; row < end; ++row) {
                uint64_t rowBase = static_cast<uint64_t>(row) * 16400;
                int32_t count = membershipGm.GetValue(rowBase + 16391);
                for (int32_t index = 0; index < count; ++index) {
                    int32_t column = membershipGm.GetValue(rowBase + index);
                    int32_t slot = outputSlotsGm.GetValue(static_cast<uint64_t>(row) * 2051 + column);
                    if (residentLocal.GetValue(slot) == 0) {
                        residentLocal.SetValue(slot, 1);
                        unionLocal.SetValue(unique++, outputIdsGm.GetValue(static_cast<uint64_t>(row) * 2051 + column));
                    }
                }
            }
        }
        PublishUnion(request, unique, owner * sourceCount);
        return true;
    }

    __aicore__ inline int32_t Token(uint32_t row, uint32_t column)
    {
        int32_t length = lengthsGm.GetValue(row);
        if (!pooled) {
            return idsGm.GetValue(static_cast<uint64_t>(row) * 2051 + column);
        }
        if (length <= 2048) {
            return static_cast<int32_t>(column) < length ? column : -1;
        }
        if (column >= 2048) {
            int32_t token = length / 4 * 4 + column - 2048;
            return token < length ? token : -1;
        }
        int32_t pool = idsGm.GetValue(static_cast<uint64_t>(row) * 512 + column / 4);
        return pool >= 0 && pool < length / 4 ? pool * 4 + column % 4 : -1;
    }

    __aicore__ inline int64_t Address(uint32_t request, uint32_t row, int32_t token)
    {
        if (token < 0 || token >= sourceCount || token >= lengthsGm.GetValue(row) || token / 128 >= tableColumns) {
            return -1;
        }
        int32_t page = tableGm.GetValue(static_cast<uint64_t>(request) * tableColumns + token / 128);
        int64_t address = static_cast<int64_t>(page) * 128 + token % 128;
        return page >= 0 && address < hostSize ? address : -1;
    }

    __aicore__ inline uint32_t SortUnion(uint32_t count)
    {
        auto keys = rowIds.ReinterpretCast<float>();
        auto indices = rowIds[1024].ReinterpretCast<uint32_t>();
        auto pairs = rowIds[2048].ReinterpretCast<float>();
        auto scratch = rowIds[4096].ReinterpretCast<float>();
        uint32_t chunks = (count + 1023) / 1024;
        uint32_t cursors[16] = {};
        uint32_t limits[16] = {};
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            uint32_t base = chunk * 1024;
            uint32_t valid = count - base < 1024 ? count - base : 1024;
            uint32_t padded = (valid + 31) / 32 * 32;
            cursors[chunk] = base;
            limits[chunk] = base + valid;
            for (uint32_t index = 0; index < padded; ++index) {
                int32_t token = index < valid ? unionLocal.GetValue(base + index) : -1;
                keys.SetValue(index, token >= 0 ? -static_cast<float>(token) : -1.0e20F);
                indices.SetValue(index, static_cast<uint32_t>(token));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID7);
            WaitFlag<HardEvent::S_V>(EVENT_ID7);
            Sort<float, true>(pairs, keys, indices, scratch, padded / 32);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            for (uint32_t index = 0; index < valid; ++index)
                unionLocal.SetValue(base + index, pairs.ReinterpretCast<int32_t>().GetValue(index * 2 + 1));
        }
        uint32_t unique = 0;
        int32_t previous = -1;
        for (uint32_t index = 0; index < count; ++index) {
            int32_t smallest = 0x7FFFFFFF;
            uint32_t selected = 0;
            for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                if (cursors[chunk] == limits[chunk]) continue;
                int32_t token = unionLocal.GetValue(cursors[chunk]);
                if (token < smallest) {
                    smallest = token;
                    selected = chunk;
                }
            }
            ++cursors[selected];
            if (smallest != previous) {
                if (unique == capacity) return capacity + 1;
                residentLocal.SetValue(unique++, smallest);
                previous = smallest;
            }
        }
        for (uint32_t index = 0; index < unique; ++index)
            unionLocal.SetValue(index, residentLocal.GetValue(index));
        return unique;
    }

    __aicore__ inline bool Less(uint32_t left, uint32_t right)
    {
        float leftScore = priorityLocal.GetValue(left);
        float rightScore = priorityLocal.GetValue(right);
        return leftScore < rightScore || (leftScore == rightScore && left < right);
    }

    __aicore__ inline void SiftVictim(uint32_t root, uint32_t count)
    {
        int32_t slot = heapLocal.GetValue(root);
        for (uint32_t child = root * 2 + 1; child < count; child = root * 2 + 1) {
            if (child + 1 < count && Less(heapLocal.GetValue(child + 1), heapLocal.GetValue(child))) ++child;
            int32_t next = heapLocal.GetValue(child);
            if (!Less(next, slot)) break;
            heapLocal.SetValue(root, next);
            root = child;
        }
        heapLocal.SetValue(root, slot);
    }

    __aicore__ inline bool SortVictims(uint32_t count)
    {
        auto keys = rowIds.ReinterpretCast<float>();
        auto indices = rowIds[1024].ReinterpretCast<uint32_t>();
        auto pairs = rowIds[2048].ReinterpretCast<float>();
        auto scratch = rowIds[4096].ReinterpretCast<float>();
        union FloatBits { uint32_t bits; float value; } negativeInfinity;
        negativeInfinity.bits = 0xFF800000U;
        for (uint32_t index = 0; index < count; ++index) {
            float score = priorityLocal.GetValue(heapLocal.GetValue(index));
            if (score != negativeInfinity.value && !(score > -1.0e30F && score < 1.0e30F)) return false;
        }
        for (uint32_t base = 0; base < count; base += 1024) {
            uint32_t valid = count - base < 1024 ? count - base : 1024;
            uint32_t padded = (valid + 31) / 32 * 32;
            bool equal = true;
            float first = priorityLocal.GetValue(heapLocal.GetValue(base));
            for (uint32_t index = 0; index < padded; ++index) {
                int32_t slot = index < valid ? heapLocal.GetValue(base + index) : -1;
                float score = slot >= 0 ? priorityLocal.GetValue(slot) : 0;
                if (slot >= 0 && score != first) equal = false;
                keys.SetValue(index, slot < 0 ? -1.0e30F :
                    (score == negativeInfinity.value ? 1.0e30F : -score));
                indices.SetValue(index, static_cast<uint32_t>(slot));
            }
            if (equal) continue;
            SetFlag<HardEvent::S_V>(EVENT_ID7);
            WaitFlag<HardEvent::S_V>(EVENT_ID7);
            Sort<float, true>(pairs, keys, indices, scratch, padded / 32);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            for (uint32_t index = 0; index < valid; ++index)
                heapLocal.SetValue(base + index, pairs.ReinterpretCast<int32_t>().GetValue(index * 2 + 1));
            for (uint32_t begin = 0; begin < valid;) {
                float score = priorityLocal.GetValue(heapLocal.GetValue(base + begin));
                uint32_t end = begin + 1;
                while (end < valid && priorityLocal.GetValue(heapLocal.GetValue(base + end)) == score) ++end;
                uint32_t tied = end - begin;
                if (tied > 1) {
                    uint32_t tiePadded = (tied + 31) / 32 * 32;
                    for (uint32_t index = 0; index < tiePadded; ++index) {
                        int32_t slot = index < tied ? heapLocal.GetValue(base + begin + index) : -1;
                        keys.SetValue(index, slot >= 0 ? -static_cast<float>(slot) : -1.0e30F);
                        indices.SetValue(index, static_cast<uint32_t>(slot));
                    }
                    SetFlag<HardEvent::S_V>(EVENT_ID7);
                    WaitFlag<HardEvent::S_V>(EVENT_ID7);
                    Sort<float, true>(pairs, keys, indices, scratch, tiePadded / 32);
                    SetFlag<HardEvent::V_S>(EVENT_ID7);
                    WaitFlag<HardEvent::V_S>(EVENT_ID7);
                    for (uint32_t index = 0; index < tied; ++index)
                        heapLocal.SetValue(base + begin + index, pairs.ReinterpretCast<int32_t>().GetValue(index * 2 + 1));
                }
                begin = end;
            }
        }
        return true;
    }

    __aicore__ inline float Priority(int32_t token, uint32_t start, uint32_t end)
    {
        union FloatBits { uint32_t bits; float value; } score;
        float total = 0;
        bool valid = false;
        if (token >= 0) {
            for (uint32_t row = start; row < end; ++row) {
                if (token / 4 >= lengthsGm.GetValue(row) / 4 || token / 4 >= scoreStride) continue;
                uint32_t bits = scoresGm.GetValue(static_cast<uint64_t>(row) * scoreStride + token / 4);
                score.bits = encoded ? ((bits & 0x80000000U) ? bits ^ 0x80000000U : ~bits) : bits;
                if ((score.bits & 0x7F800000U) == 0x7F800000U) continue;
                total += score.value;
                valid = true;
            }
        }
        score.bits = 0xFF800000U;
        return valid ? total : score.value;
    }

    __aicore__ inline void CopyRow(uint32_t row)
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID7);
        DataCopyPad(outputIdsGm[static_cast<uint64_t>(row) * 2051], rowIds, {1, 2051 * 4, 0, 0});
        DataCopyPad(outputSlotsGm[static_cast<uint64_t>(row) * 2051], rowSlots, {1, 2051 * 4, 0, 0});
        DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 16384 - 2051], rowPlan, {1, 2051 * 2, 0, 0});
        DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 16384], rowPlan[2064], {1, 22, 0, 0});
        uint32_t misses = static_cast<uint16_t>(rowPlan.GetValue(2074));
        if (misses > 0 && misses < 2048)
            DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 4096],
                rowIds[6240].ReinterpretCast<int16_t>(), DataCopyExtParams{1, misses * 2, 0, 0, 0});
        if (misses > 0 && misses < 2048)
            DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 4096 + misses],
                rowIds[7280].ReinterpretCast<int16_t>(), DataCopyExtParams{1, (2048 - misses) * 2, 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
    }

    __aicore__ inline void ProcessRequest(uint32_t request)
    {
        int32_t start = request == 0 ? 0 : endsGm.GetValue(request - 1);
        int32_t end = endsGm.GetValue(request);
        int64_t owner = requestsGm.GetValue(request);
        if (start < 0 || end < start || end > rowCount || end - start > 6 || (end - start) * 2051 > unionSize) return;
        bool active = owner > 0 && owner < requestCount;
        uint64_t residentBase = active ? owner * capacity : 0;
        uint64_t mapBase = active ? owner * sourceCount : 0;
        uint32_t total = 0;
        int32_t maxLength = 0;
        for (int32_t row = start; row < end; ++row) {
            int32_t length = lengthsGm.GetValue(row);
            maxLength = maxLength > length ? maxLength : length;
        }
        if (FinishPrepared(request, start, end, owner, maxLength)) return;
        bool allMapped = active && stateGm.GetValue(owner * 16) == generationsGm.GetValue(owner) &&
            maxLength >= stateGm.GetValue(owner * 16 + 1);
        if (active) stateGm.SetValue(owner * 16 + 2, !allMapped &&
            (stateGm.GetValue(owner * 16) != generationsGm.GetValue(owner) ||
             maxLength < stateGm.GetValue(owner * 16 + 1)));
        if (allMapped) {
            for (uint32_t slot = 0; slot < capacity; ++slot) residentLocal.SetValue(slot, -1);
        }
        for (int32_t row = start; row < end; ++row) {
            if (!active) continue;
            for (uint32_t column = 0; column < 2051; ++column) {
                int32_t token = Token(row, column);
                if (Address(request, row, token) >= 0) {
                    unionLocal.SetValue(total++, token);
                    if (allMapped) {
                        int32_t slot = mappingGm.GetValue(mapBase + token);
                        if (slot >= 0 && slot < capacity) residentLocal.SetValue(slot, token);
                        else allMapped = false;
                    }
                }
            }
        }
        uint32_t unique = 0;
        if (allMapped) {
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                int32_t token = residentLocal.GetValue(slot);
                if (token >= 0) unionLocal.SetValue(unique++, token);
            }
            stateGm.SetValue(owner * 16 + 1, maxLength);
        } else {
            unique = SortUnion(total);
        }
        if (unique > capacity) return;
        if (active && !allMapped) {
            int64_t generation = generationsGm.GetValue(owner);
            bool reset = stateGm.GetValue(owner * 16) != generation || maxLength < stateGm.GetValue(owner * 16 + 1);
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                int32_t token = sourcesGm.GetValue(residentBase + slot);
                if (reset) {
                    if (token >= 0 && token < sourceCount) mappingGm.SetValue(mapBase + token, -1);
                    sourcesGm.SetValue(residentBase + slot, -1);
                    physicalGm.SetValue(residentBase + slot, -1);
                    token = -1;
                }
                residentLocal.SetValue(slot, token);
            }
            uint32_t missing = 0;
            for (uint32_t index = 0; index < unique; ++index) {
                int32_t token = unionLocal.GetValue(index);
                int32_t slot = mappingGm.GetValue(mapBase + token);
                if (slot >= 0 && slot < capacity) residentLocal.SetValue(slot, -token - 2);
                else ++missing;
            }
            uint32_t heapCount = 0;
            if (missing > 0) {
                for (uint32_t slot = 0; slot < capacity; ++slot) {
                    int32_t token = residentLocal.GetValue(slot);
                    if (token < -1) continue;
                    heapLocal.SetValue(heapCount++, slot);
                    priorityLocal.SetValue(slot, Priority(token, start, end));
                }
                bool sortedVictims = !reset && SortVictims(heapCount);
                uint32_t victimChunks = (heapCount + 1023) / 1024;
                uint32_t victimCursors[16] = {};
                for (uint32_t chunk = 0; chunk < victimChunks; ++chunk) victimCursors[chunk] = chunk * 1024;
                if (!reset && !sortedVictims)
                    for (int32_t root = static_cast<int32_t>(heapCount / 2) - 1; root >= 0; --root) SiftVictim(root, heapCount);
                uint32_t freeCursor = 0;
                for (uint32_t index = 0; index < unique; ++index) {
                    int32_t token = unionLocal.GetValue(index);
                    int32_t slot = mappingGm.GetValue(mapBase + token);
                    if (slot >= 0 && slot < capacity) continue;
                    if (reset) slot = heapLocal.GetValue(freeCursor++);
                    else if (sortedVictims) {
                        int32_t selected = -1;
                        for (uint32_t chunk = 0; chunk < victimChunks; ++chunk) {
                            uint32_t cursor = victimCursors[chunk];
                            if (cursor >= heapCount || cursor >= (chunk + 1) * 1024) continue;
                            int32_t candidate = heapLocal.GetValue(cursor);
                            if (selected < 0 || Less(candidate, slot)) {
                                slot = candidate;
                                selected = chunk;
                            }
                        }
                        ++victimCursors[selected];
                    }
                    else {
                        slot = heapLocal.GetValue(0);
                        heapLocal.SetValue(0, heapLocal.GetValue(--heapCount));
                        if (heapCount) SiftVictim(0, heapCount);
                    }
                    int32_t old = residentLocal.GetValue(slot);
                    if (old >= 0) mappingGm.SetValue(mapBase + old, -1);
                    mappingGm.SetValue(mapBase + token, slot);
                    sourcesGm.SetValue(residentBase + slot, token);
                    physicalGm.SetValue(residentBase + slot, -1);
                }
            }
            stateGm.SetValue(owner * 16, generation);
            stateGm.SetValue(owner * 16 + 1, maxLength);
        }
        PublishUnion(request, unique, mapBase);
        for (int32_t row = start; row < end; ++row) {
            uint32_t mutablePresence = 0;
            uint32_t bodyMisses = 0;
            uint32_t bodyHits = 0;
            auto missOrder = rowIds[6240].ReinterpretCast<int16_t>();
            auto hitOrder = rowIds[7280].ReinterpretCast<int16_t>();
            for (uint32_t column = 0; column < 2051; ++column) {
                int32_t token = active ? Token(row, column) : -1;
                int64_t address = Address(request, row, token);
                int32_t slot = address >= 0 ? mappingGm.GetValue(mapBase + token) : -1;
                bool valid = slot >= 0 && slot < capacity;
                bool hit = valid && physicalGm.GetValue(residentBase + slot) == address &&
                    token < lengthsGm.GetValue(row) - static_cast<int32_t>(tailSize);
                int32_t tailOffset = token - lengthsGm.GetValue(row) + static_cast<int32_t>(tailSize);
                if (valid && tailOffset >= 0 && tailOffset < 32)
                    mutablePresence |= 1U << tailOffset;
                rowIds.SetValue(column, valid ? token : -1);
                rowSlots.SetValue(column, valid ? slot : -1);
                rowPlan.SetValue(column, valid ? (hit ? slot + 1 : -slot - 1) : 0);
                if (column < 2048 && valid && !hit) missOrder.SetValue(bodyMisses++, column);
            }
            if (bodyMisses > 0 && bodyMisses < 2048)
                for (uint32_t column = 0; column < 2048; ++column)
                    if (rowPlan.GetValue(column) >= 0) hitOrder.SetValue(bodyHits++, column);
            rowPlan.SetValue(2064, 0x5A4D);
            rowPlan.SetValue(2065, 0x5A45);
            rowPlan.SetValue(2066, 2051);
            rowPlan.SetValue(2067, 16384 - 2051);
            rowPlan.SetValue(2068, active ? owner : 0);
            rowPlan.SetValue(2069, 0x5A44);
            rowPlan.SetValue(2070, capacity);
            rowPlan.SetValue(2071, 0);
            rowPlan.SetValue(2072, static_cast<int16_t>(mutablePresence & 0xFFFFU));
            rowPlan.SetValue(2073, static_cast<int16_t>(mutablePresence >> 16));
            rowPlan.SetValue(2074, bodyMisses);
            CopyRow(row);
        }
    }

    GlobalTensor<int32_t> idsGm, endsGm, lengthsGm, tableGm, mappingGm, sourcesGm, outputIdsGm, outputSlotsGm;
    GlobalTensor<int32_t> unionSourcesGm, unionSlotsGm;
    GlobalTensor<uint32_t> scoresGm;
    GlobalTensor<int64_t> requestsGm, generationsGm, physicalGm, stateGm, unionPhysicalGm;
    GlobalTensor<int16_t> membershipGm;
    TBuf<TPosition::VECCALC> unionBuffer, residentBuffer, heapBuffer, priorityBuffer, rowBuffer;
    LocalTensor<int32_t> unionLocal, residentLocal, heapLocal, rowIds, rowSlots;
    LocalTensor<int16_t> rowPlan;
    LocalTensor<float> priorityLocal;
    uint32_t rowCount, batchCount, requestCount, sourceCount, capacity, tableColumns, unionSize, scoreStride, tailSize;
    uint64_t hostSize;
    bool pooled, encoded, prepared;
};
}
