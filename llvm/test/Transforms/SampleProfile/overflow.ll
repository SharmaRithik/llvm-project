; NOTE: Assertions have been autogenerated by utils/update_analyze_test_checks.py UTC_ARGS: --version 4

; Checks that we are able to handle overflowing counters correctly.

; RUN: opt < %s -passes='sample-profile,print<branch-prob>' -sample-profile-file=%S/Inputs/overflow.proftext -disable-output 2>&1 | FileCheck %s

; Original Source:
; int sqrt(int);
; int test(int i) {
;    if (i == 5) {
;        return 42;
;    }
;    else {
;        return sqrt(i);
;    }
;}

define dso_local noundef i32 @_Z3testi(i32 noundef %i) local_unnamed_addr #0 !dbg !10 {
; CHECK-LABEL: '_Z3testi'
; CHECK-NEXT:  ---- Branch Probabilities ----
; CHECK-NEXT:    edge %entry -> %return probability is 0x00000000 / 0x80000000 = 0.00%
; CHECK-NEXT:    edge %entry -> %if.else probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; CHECK-NEXT:    edge %if.else -> %return probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;
entry:
  tail call void @llvm.dbg.value(metadata i32 %i, metadata !16, metadata !DIExpression()), !dbg !17
  %cmp = icmp eq i32 %i, 5, !dbg !18
  br i1 %cmp, label %return, label %if.else, !dbg !20

if.else:                                          ; preds = %entry
  %call = tail call noundef i32 @_Z4sqrti(i32 noundef %i), !dbg !21
  br label %return, !dbg !23

return:                                           ; preds = %entry, %if.else
  %retval.0 = phi i32 [ %call, %if.else ], [ 42, %entry ], !dbg !24
  ret i32 %retval.0, !dbg !25
}

declare !dbg !26 noundef i32 @_Z4sqrti(i32 noundef)

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "cb38d90153a7ebdd6ecf3058eb0524c7")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "test", linkageName: "_Z3loli", scope: !11, file: !11, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!11 = !DIFile(filename: "./test.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "cb38d90153a7ebdd6ecf3058eb0524c7")
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DILocalVariable(name: "i", arg: 1, scope: !10, file: !11, line: 3, type: !14)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocation(line: 4, column: 11, scope: !19)
!19 = distinct !DILexicalBlock(scope: !10, file: !11, line: 4, column: 9)
!20 = !DILocation(line: 4, column: 9, scope: !10)
!21 = !DILocation(line: 8, column: 16, scope: !22)
!22 = distinct !DILexicalBlock(scope: !19, file: !11, line: 7, column: 10)
!23 = !DILocation(line: 8, column: 9, scope: !22)
!24 = !DILocation(line: 0, scope: !19)
!25 = !DILocation(line: 10, column: 1, scope: !10)
!26 = !DISubprogram(name: "sqrt", linkageName: "_Z4sqrti", scope: !11, file: !11, line: 1, type: !12, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)

