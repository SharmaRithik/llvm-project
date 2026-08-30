// REQUIRES: asserts
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -O1 \
// RUN:   -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir \
// RUN:   -mmlir -mlir-pass-statistics -mmlir \
// RUN:   -mlir-pass-statistics-display=list %s -o /dev/null 2> %t.stats
// RUN: FileCheck %s --check-prefix=STATS --input-file=%t.stats

namespace std {
template <class Iter1, class Iter2>
Iter1 search(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);
template <class T1, class T2> struct pair {
  T1 first;
  T2 second;
};
template <class Iter1, class Iter2>
pair<Iter1, Iter2> mismatch(Iter1 first1, Iter1 last1, Iter2 first2);
template <class Iter1, class Iter2>
pair<Iter1, Iter2> mismatch(Iter1 first1, Iter1 last1, Iter2 first2,
                            Iter2 last2);
template <class Iter1, class Iter2, class Pred>
pair<Iter1, Iter2> mismatch(Iter1 first1, Iter1 last1, Iter2 first2,
                            Pred pred) {
  for (; first1 != last1; ++first1, ++first2)
    if (!pred(*first1, *first2))
      break;
  return {first1, first2};
}
template <class Iter1, class Iter2, class Pred>
pair<Iter1, Iter2> mismatch(Iter1 first1, Iter1 last1, Iter2 first2,
                            Iter2 last2, Pred pred) {
  for (; first1 != last1 && first2 != last2; ++first1, ++first2)
    if (!pred(*first1, *first2))
      break;
  return {first1, first2};
}
}

unsigned char *rewritten_search(unsigned char *first1, unsigned char *last1,
                                unsigned char *first2,
                                unsigned char *last2) {
  return std::search(first1, last1, first2, last2);
}

short *declined_search(short *first1, short *last1, short *first2,
                       short *last2) {
  return std::search(first1, last1, first2, last2);
}

int *rewritten_wide_search(int *first1, int *last1, int *first2,
                           int *last2) {
  return std::search(first1, last1, first2, last2);
}

std::pair<char *, char *> mismatch_one(char *first1, char *last1,
                                       char *first2) {
  return std::mismatch(first1, last1, first2);
}

std::pair<char *, char *> mismatch_bounded_one(char *first1, char *last1,
                                               char *first2, char *last2) {
  return std::mismatch(first1, last1, first2, last2);
}

std::pair<char *, char *> mismatch_bounded_two(char *first1, char *last1,
                                               char *first2, char *last2) {
  return std::mismatch(first1, last1, first2, last2);
}

std::pair<char *, char *> mismatch_pred_one(char *first1, char *last1,
                                            char *first2) {
  return std::mismatch(first1, last1, first2,
                       [](char lhs, char rhs) { return lhs == rhs; });
}

std::pair<char *, char *> mismatch_pred_two(char *first1, char *last1,
                                            char *first2) {
  return std::mismatch(first1, last1, first2,
                       [](char lhs, char rhs) { return lhs == rhs; });
}

std::pair<char *, char *> mismatch_pred_three(char *first1, char *last1,
                                              char *first2) {
  return std::mismatch(first1, last1, first2,
                       [](char lhs, char rhs) { return lhs == rhs; });
}

std::pair<char *, char *>
mismatch_bounded_pred_one(char *first1, char *last1, char *first2,
                          char *last2) {
  return std::mismatch(first1, last1, first2, last2,
                       [](char lhs, char rhs) { return lhs == rhs; });
}

std::pair<char *, char *>
mismatch_bounded_pred_two(char *first1, char *last1, char *first2,
                          char *last2) {
  return std::mismatch(first1, last1, first2, last2,
                       [](char lhs, char rhs) { return lhs == rhs; });
}

std::pair<char *, char *>
mismatch_bounded_pred_three(char *first1, char *last1, char *first2,
                            char *last2) {
  return std::mismatch(first1, last1, first2, last2,
                       [](char lhs, char rhs) { return lhs == rhs; });
}

std::pair<char *, char *>
mismatch_bounded_pred_four(char *first1, char *last1, char *first2,
                           char *last2) {
  return std::mismatch(first1, last1, first2, last2,
                       [](char lhs, char rhs) { return lhs == rhs; });
}

// STATS: IdiomRecognizer
// STATS: (S) 1 raised-std-mismatch
// STATS-NEXT: (S) 2 raised-std-mismatch-bounded
// STATS-NEXT: (S) 4 raised-std-mismatch-bounded-pred
// STATS-NEXT: (S) 3 raised-std-mismatch-pred
// STATS: (S) 3 raised-std-search
// STATS: LibOpt
// STATS: (S) 1 search-equal-length-to-memcmp
// STATS: (S) 1 search-to-memmem
// STATS: (S) 1 search-wide-to-wmemchr
