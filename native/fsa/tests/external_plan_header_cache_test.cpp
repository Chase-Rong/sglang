#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

void Check(bool condition, const char *expression, int line) {
  if (!condition) {
    throw std::runtime_error(std::string("line ") + std::to_string(line) +
                             ": " + expression);
  }
}

#define CHECK(expression) Check((expression), #expression, __LINE__)
#define __gm__

struct Transfer {
  int64_t offset;
  int64_t count;
  bool header;
};

struct Observations {
  int64_t stride = 16400;
  int subblock = 0;
  int sets = 0;
  int waits = 0;
  std::vector<Transfer> copies;

  int Headers() const {
    return static_cast<int>(
        std::count_if(copies.begin(), copies.end(),
                      [](const Transfer &copy) { return copy.header; }));
  }

  int Slices() const { return static_cast<int>(copies.size()) - Headers(); }
} observed;

template <typename Value> struct LocalTensor {
  Value *data;
  int64_t size;

  Value GetValue(int64_t index) const {
    CHECK(index >= 0 && index < size);
    return data[index];
  }
};

template <typename Value> struct GlobalTensor {
  Value *data = nullptr;
  int64_t size = 0;
  int64_t offset = 0;
  int writes = 0;
  mutable int reads = 0;

  void SetGlobalBuffer(Value *pointer) {
    data = pointer;
    offset = 0;
  }

  GlobalTensor operator[](int64_t index) const {
    CHECK(index >= 0 && index < size);
    return {data + index, size - index, offset + index};
  }

  Value GetValue(int64_t index) const {
    CHECK(data != nullptr && index >= 0 && index < size);
    ++reads;
    return data[index];
  }

  void SetValue(int64_t index, Value value) {
    CHECK(data != nullptr && index >= 0 && index < size);
    data[index] = value;
    ++writes;
  }
};

struct TBuf {
  std::array<int16_t, 256> storage{};

  template <typename Value> LocalTensor<Value> Get() {
    static_assert(std::is_same_v<Value, int16_t>);
    return {storage.data(), static_cast<int64_t>(storage.size())};
  }
};

struct DataCopyExtParams {
  uint32_t blockCount = 0, blockLen = 0, srcStride = 0, dstStride = 0;
};

template <typename Value> struct DataCopyPadExtParams {
  bool isPad = false;
  uint8_t leftPadding = 0, rightPadding = 0;
  Value paddingValue = 0;
};

void DataCopyPad(LocalTensor<int16_t> destination, GlobalTensor<int16_t> source,
                 const DataCopyExtParams &params,
                 const DataCopyPadExtParams<int16_t> &padding) {
  CHECK(params.blockCount == 1 && params.srcStride == 0 &&
        params.dstStride == 0);
  CHECK(!padding.isPad && padding.leftPadding == 0 &&
        padding.rightPadding == 0);
  CHECK(params.blockLen % sizeof(int16_t) == 0);
  int64_t count = params.blockLen / sizeof(int16_t);
  CHECK(source.data != nullptr && count <= source.size &&
        count <= destination.size);
  bool header = source.offset % observed.stride == 16384;
  CHECK(!header || count == 8);
  observed.copies.push_back({source.offset, count, header});
  std::copy_n(source.data, count, destination.data);
}

enum class HardEvent { MTE2_S };
template <HardEvent> void SetFlag(int) { ++observed.sets; }
template <HardEvent> void WaitFlag(int) { ++observed.waits; }
int GetSubBlockIdx() { return observed.subblock; }

struct RunInfo {
  int64_t boIdx = 0, s1oIdx = 0, n2oIdx = 0, s2LoopCount = 0, s2RealSize = 128;
};

struct ConstInfo {
  int64_t s2BaseSize = 128, s1Size = 2, n2Size = 1;
};

struct Tiling {
  struct {
    int64_t blockSize = 128, selectionBlockTableStride = 8,
            selectionMembershipStride = 16400;
  } baseParams;
};

enum class FusedSparseAttentionOverlapLayoutArch35 { TND, BSND };

template <FusedSparseAttentionOverlapLayoutArch35 LAYOUT_T> struct Parser {
  using KV_T = int16_t;
  static constexpr bool isPa = true;
  Tiling *tilingData = nullptr;
  TBuf selectionPlanTBuf;
  int mte2ToSPlan = 0;
  GlobalTensor<KV_T> keyGm, keyRopeGm, selectionKRopeGm, selectionKvCacheGm;
  GlobalTensor<int32_t> blockTableGm, sparseIndicesGm, selectionKvBlockTableGm;
  GlobalTensor<int32_t> selectionKvBlockStatusGm, selectionKvActualSeqGm,
      cuSeqlensQGm;
  GlobalTensor<int16_t> selectionMembershipMapGm;

#include "external_plan_header_cache_extracted.inc"
};

template <FusedSparseAttentionOverlapLayoutArch35 Layout =
              FusedSparseAttentionOverlapLayoutArch35::TND>
struct Fixture {
  static constexpr int64_t rows = 8;
  Tiling tiling;
  std::vector<int16_t> membership = std::vector<int16_t>(rows * 16400);
  std::array<int32_t, rows> actualSeq;
  std::array<int32_t, 3> queryEnds{2, 4, 8};
  std::array<int16_t, 128> slice{};
  Parser<Layout> parser;
  ConstInfo info;
  RunInfo run;
  int64_t logicalRow = -99;

  Fixture() {
    observed = {};
    actualSeq.fill(-99);
    parser.tilingData = &tiling;
    parser.selectionMembershipMapGm.size =
        static_cast<int64_t>(membership.size());
    parser.selectionKvActualSeqGm.size = rows;
    parser.cuSeqlensQGm.size = static_cast<int64_t>(queryEnds.size());
    parser.cuSeqlensQGm.SetGlobalBuffer(queryEnds.data());
    Init();
  }

  void Init() {
    observed.stride = tiling.baseParams.selectionMembershipStride;
    parser.InitGlobalBuffer(nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr,
                            reinterpret_cast<uint8_t *>(membership.data()),
                            reinterpret_cast<uint8_t *>(actualSeq.data()));
  }

  int16_t *Header(int64_t row) {
    return membership.data() + row * observed.stride + 16384;
  }

  void Publish(int64_t row, int16_t count = 2176, int16_t owner = 3,
               int16_t stride = 256) {
    CHECK(row >= 0 && row < rows && count >= 0 && count <= 16384);
    std::array<int16_t, 8> header{
        0x5A4D, 0x5A45, count,  static_cast<int16_t>(16384 - count),
        owner,  0x5A53, stride, 3};
    std::copy(header.begin(), header.end(), Header(row));
    auto start = membership.begin() + row * observed.stride + 16384 - count;
    for (int64_t index = 0; index < count; ++index) {
      start[index] = static_cast<int16_t>(index % 2 == 0 ? index % 127 + 1
                                                         : -(index % 127 + 1));
    }
  }

  bool Prepare() {
    slice.fill(-30000);
    bool accepted = parser.PrepareExternalSelectionPlan(
        run, info, {slice.data(), static_cast<int64_t>(slice.size())},
        logicalRow);
    CHECK(observed.sets == static_cast<int>(observed.copies.size()));
    CHECK(observed.waits == observed.sets);
    CHECK(parser.selectionMembershipMapGm.reads == 0);
    if (accepted) {
      CHECK(parser.externalSelectionPlanActive);
      int64_t offset = logicalRow * observed.stride +
                       parser.selectionPlanOffset +
                       run.s2LoopCount * info.s2BaseSize;
      CHECK(std::equal(slice.begin(), slice.begin() + run.s2RealSize,
                       membership.begin() + offset));
      CHECK(!observed.copies.back().header &&
            observed.copies.back().offset == offset);
      CHECK(observed.copies.back().count == run.s2RealSize);
    } else {
      CHECK(!parser.externalSelectionPlanActive);
      CHECK(slice.front() == -30000);
    }
    return accepted;
  }

  void Decoded(int64_t row, int64_t count, int64_t owner, int64_t stride) {
    CHECK(parser.selectionPreparedRow == row &&
          parser.selectionPreparedCount == count);
    CHECK(parser.selectionDataRow == owner &&
          parser.selectionDirectRowStride == stride);
    CHECK(parser.selectionPlanOffset == 16384 - count);
  }

  void ResetState() {
    CHECK(!parser.externalSelectionPlanActive);
    CHECK(parser.selectionPreparedRow == -1 &&
          parser.selectionPreparedCount == 0);
    CHECK(parser.selectionDataRow == -1 &&
          parser.selectionDirectRowStride == 0);
    CHECK(parser.selectionPlanOffset == 0);
  }
};

void SameRowTiles() {
  Fixture<> fixture;
  fixture.Publish(0, 2051);
  for (int tile = 0; tile < 17; ++tile) {
    fixture.run.s2LoopCount = tile;
    fixture.run.s2RealSize = std::min<int64_t>(128, 2051 - tile * 128);
    CHECK(fixture.Prepare());
    fixture.Decoded(0, 2051, 3, 256);
  }
  CHECK(observed.Headers() == 1 && observed.Slices() == 17);
  CHECK(observed.copies.back().count == 3);
  CHECK(fixture.actualSeq[0] == 2051 &&
        fixture.parser.selectionKvActualSeqGm.writes == 1);
}

void FreshSignedSlices() {
  Fixture<> fixture;
  fixture.Publish(0);
  for (int pass = 0; pass < 17; ++pass) {
    fixture.run.s2LoopCount = pass % 3;
    int64_t offset = 16384 - 2176 + fixture.run.s2LoopCount * 128;
    for (int index = 0; index < 128; ++index) {
      fixture.membership[offset + index] = static_cast<int16_t>(
          index % 3 == 0 ? 0 : (index % 2 == 0 ? pass + 1 : -(pass + 1)));
    }
    CHECK(fixture.Prepare());
  }
  CHECK(observed.Headers() == 1 && observed.Slices() == 17);
}

template <FusedSparseAttentionOverlapLayoutArch35 Layout> void RowRefresh() {
  Fixture<Layout> fixture;
  fixture.queryEnds[0] = 3;
  int64_t nextRequestRow =
      Layout == FusedSparseAttentionOverlapLayoutArch35::TND ? 3 : 2;
  fixture.Publish(0, 256, 3, 64);
  fixture.Publish(1, 384, 3, 128);
  fixture.Publish(nextRequestRow, 512, 7, 256);
  CHECK(fixture.Prepare());
  fixture.Decoded(0, 256, 3, 64);
  fixture.run.s1oIdx = 1;
  CHECK(fixture.Prepare());
  fixture.Decoded(1, 384, 3, 128);
  fixture.run.boIdx = 1;
  fixture.run.s1oIdx = 0;
  CHECK(fixture.Prepare());
  fixture.Decoded(nextRequestRow, 512, 7, 256);
  fixture.run.boIdx = 0;
  CHECK(fixture.Prepare());
  fixture.Decoded(0, 256, 3, 64);
  CHECK(observed.Headers() == 4 && observed.Slices() == 4);
}

void HeadRowRefresh() {
  Fixture<> fixture;
  fixture.info.n2Size = 2;
  fixture.Publish(0, 256, 3, 64);
  fixture.Publish(1, 256, 4, 128);
  CHECK(fixture.Prepare());
  fixture.run.n2oIdx = 1;
  CHECK(fixture.Prepare());
  fixture.Decoded(1, 256, 4, 128);
  CHECK(observed.Headers() == 2 && observed.Slices() == 2);
}

void ValidBoundaries() {
  Fixture<> fixture;
  fixture.Publish(0, 128, 0, 1);
  CHECK(fixture.Prepare());
  fixture.Decoded(0, 128, 0, 1);
  fixture.Publish(1, 16384, 32767, 32767);
  fixture.run.s1oIdx = 1;
  fixture.run.s2LoopCount = 127;
  CHECK(fixture.Prepare());
  fixture.Decoded(1, 16384, 32767, 32767);
  CHECK(observed.Headers() == 2 && observed.Slices() == 2);
}

void PreDmaRejection(const std::string &name) {
  Fixture<> fixture;
  fixture.Publish(0);
  CHECK(fixture.Prepare());
  if (name == "negative-row") {
    fixture.run.s1oIdx = -1;
  } else {
    fixture.run.s1oIdx = 1;
    fixture.parser.selectionMembershipStride =
        name == "short-membership" ? 16399 : 0;
  }
  CHECK(!fixture.Prepare());
  fixture.ResetState();
  CHECK(observed.Headers() == 1 && observed.Slices() == 1);
}

void InvalidHeader(int field, int16_t value) {
  Fixture<> fixture;
  fixture.Publish(0);
  CHECK(fixture.Prepare());
  fixture.Publish(1);
  fixture.Header(1)[field] = value;
  fixture.run.s1oIdx = 1;
  CHECK(!fixture.Prepare());
  fixture.ResetState();
  CHECK(observed.Headers() == 2 && observed.Slices() == 1);
  CHECK(fixture.actualSeq[1] == -99);
  fixture.Publish(1, 256, 0, 1);
  CHECK(fixture.Prepare());
  fixture.Decoded(1, 256, 0, 1);
  CHECK(observed.Headers() == 3 && observed.Slices() == 2);
}

void ShortCachedPlan() {
  Fixture<> fixture;
  fixture.Publish(0, 129);
  CHECK(fixture.Prepare());
  fixture.run.s2LoopCount = 1;
  fixture.run.s2RealSize = 1;
  CHECK(fixture.Prepare());
  fixture.run.s2RealSize = 2;
  CHECK(!fixture.Prepare());
  CHECK(observed.Headers() == 1 && observed.Slices() == 2);
  CHECK(fixture.parser.selectionDataRow == -1 &&
        fixture.parser.selectionDirectRowStride == 0);
  CHECK(fixture.parser.selectionPlanOffset == 0);
  CHECK(fixture.parser.selectionPreparedRow == 0 &&
        fixture.parser.selectionPreparedCount == 129);
  CHECK(fixture.parser.selectionKvActualSeqGm.writes == 1);
  fixture.Publish(0, 256, 6, 512);
  CHECK(fixture.Prepare());
  fixture.Decoded(0, 256, 6, 512);
  CHECK(observed.Headers() == 2 && observed.Slices() == 3);
  CHECK(fixture.actualSeq[0] == 129);
}

void LaunchReset(bool rebind) {
  Fixture<> fixture;
  fixture.Publish(0, 256);
  CHECK(fixture.Prepare());
  if (rebind) {
    fixture.tiling.baseParams.selectionMembershipStride = 16432;
    fixture.membership = std::vector<int16_t>(Fixture<>::rows * 16432);
    fixture.parser.selectionMembershipMapGm.size =
        static_cast<int64_t>(fixture.membership.size());
  }
  for (int launch = 0; launch < 3; ++launch) {
    fixture.Init();
    fixture.ResetState();
    int16_t count = static_cast<int16_t>(384 + launch * 128);
    fixture.Publish(0, count, static_cast<int16_t>(7 + launch),
                    static_cast<int16_t>(64 << launch));
    CHECK(fixture.Prepare());
    fixture.Decoded(0, count, 7 + launch, 64 << launch);
    CHECK(observed.Headers() == launch + 2 && observed.Slices() == launch + 2);
  }
}

void ActualSeqGate(int subblock, int tile) {
  Fixture<> fixture;
  fixture.Publish(0, 256);
  observed.subblock = subblock;
  fixture.run.s2LoopCount = tile;
  CHECK(fixture.Prepare());
  CHECK(fixture.Prepare());
  bool updates = subblock == 0 && tile == 0;
  CHECK(fixture.actualSeq[0] == (updates ? 256 : -99));
  CHECK(fixture.parser.selectionKvActualSeqGm.writes == (updates ? 2 : 0));
  CHECK(observed.Headers() == 1 && observed.Slices() == 2);
}

int main(int argc, char **argv) {
  try {
    CHECK(argc >= 2);
    std::string name = argv[1];
    if (name == "same-row-17-tiles")
      SameRowTiles();
    else if (name == "fresh-signed-slices")
      FreshSignedSlices();
    else if (name == "row-refresh-tnd")
      RowRefresh<FusedSparseAttentionOverlapLayoutArch35::TND>();
    else if (name == "row-refresh-bsnd")
      RowRefresh<FusedSparseAttentionOverlapLayoutArch35::BSND>();
    else if (name == "head-row-refresh")
      HeadRowRefresh();
    else if (name == "valid-boundaries")
      ValidBoundaries();
    else if (name == "negative-row" || name == "short-membership" ||
             name == "zero-membership")
      PreDmaRejection(name);
    else if (name == "invalid-header") {
      CHECK(argc == 4);
      InvalidHeader(std::stoi(argv[2]),
                    static_cast<int16_t>(std::stoi(argv[3])));
    } else if (name == "short-cached-plan")
      ShortCachedPlan();
    else if (name == "new-launch")
      LaunchReset(true);
    else if (name == "graph-replay-reset")
      LaunchReset(false);
    else if (name == "actual-seq") {
      CHECK(argc == 4);
      ActualSeqGate(std::stoi(argv[2]), std::stoi(argv[3]));
    } else
      throw std::runtime_error("unknown case: " + name);
    std::cout << "PASS " << name
              << " (CPU extracted parser; no NPU execution)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
