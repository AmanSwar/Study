# MODULE 1: The Execution Model Every Programmer Must Internalize

## 1. CONCEPTUAL FOUNDATION

### The End-to-End Compilation Pipeline
To understand how your ML inference code actually executes, you must trace the complete path from source to machine code to process execution. This is not trivia—the design decisions embedded in this pipeline directly enable or prevent the performance optimizations you'll rely on.

**Compilation Stages (Reference: CS:APP Chapter 1.2, "The Hello World Program")**

The C/C++ compilation process consists of four distinct stages:

1. **Preprocessing**: The C preprocessor (cpp) performs textual transformations:
   - Macro expansion: `#define MAX 1024` becomes literal substitution
   - File inclusion: `#include <stdio.h>` textually inserts header content
   - Conditional compilation: `#ifdef DEBUG` removes code at text level
   - Output: `.i` file (still human-readable C code)

2. **Compilation**: The compiler (cc1 for GCC) transforms C into assembly:
   - Lexical analysis: tokenization (identifier, keyword, number, operator)
   - Syntax analysis: parsing into abstract syntax tree (AST)
   - Semantic analysis: type checking, symbol resolution
   - Intermediate representation (IR): frontend-independent code form
   - Optimization passes: SSA form, dead code elimination, loop unrolling, inlining
   - Code generation: IR → assembly (target-specific, instruction selection)
   - Output: `.s` file (human-readable assembly)

3. **Assembly**: The assembler (as) converts mnemonics to machine code:
   - Mnemonic resolution: `movq %rax, %rbx` → `0x48 0x89 0xd8`
   - Symbol table creation: labels `_main:` → offset 0x1000
   - Relocation records: for unresolved external symbols
   - Output: `.o` file (object code, non-executable)

4. **Linking**: The linker (ld) combines object files into executable:
   - Symbol resolution: unresolved references matched across `.o` files
   - Relocation application: adjust addresses for final load location
   - Library inclusion: static (libm.a) or dynamic (libc.so.6) linking
   - Output: executable ELF binary

**Key Insight**: Optimizations are NOT uniform across stages. A compiler pass at -O2 might miss an optimization that linking-time interprocedural analysis (LTIPO) catches. Understanding which passes run where is critical for writing code that compilers can optimize.

### The ELF Format: How Your Binary is Structured

The Executable and Linkable Format (ELF) is the standard binary format on Linux. Its structure directly impacts how the OS loads your process and determines the memory layout you'll work with.

**ELF Header** (64-byte structure on x86-64):
```
Offset  Size  Field
0x00    4     Magic number: 0x7f 'E' 'L' 'F'
0x04    1     ELF class: 1=32-bit, 2=64-bit
0x05    1     Data encoding: 1=little-endian, 2=big-endian
0x06    1     ELF version
0x07    1     OS/ABI identification
0x08    8     Padding
0x10    2     Object file type: 2=executable, 3=shared object
0x12    2     Machine type: 0x3e=x86-64
0x14    4     Object file version
0x18    8     Entry point virtual address
0x20    8     Program header offset
0x28    8     Section header offset
0x30    4     Flags
0x34    2     ELF header size: 64 bytes
0x36    2     Program header entry size: 56 bytes
```

**Program Headers** (56-byte entries describing runtime segments):
```
Type        Value  Meaning
PT_LOAD     1      Loadable segment (maps to memory during execution)
PT_DYNAMIC  3      Dynamic linking info (for runtime linker)
PT_INTERP   3      Path to dynamic linker (e.g., /lib64/ld-linux-x86-64.so.2)
PT_NOTE     4      Auxiliary info (build version, capabilities)
PT_GNU_EH_FRAME 0x6474e550  Exception handling frame info
```

**Section Headers** (64-byte entries describing static sections for linking):
```
.text       Machine code (read-only, executable)
.rodata     Read-only data (constants, string literals)
.data       Initialized global/static variables (writable)
.bss        Zero-initialized globals (marked present, not stored in binary)
.symtab     Symbol table (function/variable names and offsets)
.strtab     String table (null-terminated symbol names)
.reloc.*    Relocation entries for the linker
.debug_*    Debugging symbols (stripped in release builds)
```

**Critical Performance Detail**: The `.bss` section is not stored in the binary file—it's a directive to the loader to allocate zeroed memory. A 1GB BSS (e.g., `static char buffer[1UL << 30];`) adds almost nothing to binary size but significant startup time if the kernel isn't using lazy page allocation.

### From Binary to Running Process: The Kernel's Role

When you execute `./my_inference_engine`, the kernel performs these steps:

**1. Process Creation**
- `execve()` syscall: replaces current process image
- Kernel validates ELF header, checks execute permission
- Allocates virtual address space

**2. Virtual Address Space Layout (x86-64, 48-bit canonical form)**
```
Address Range           Segment         Usage
0xffffffffffffffff      Kernel space    (inaccessible from user mode)
0xfffffffffff00000      Kernel heap
...                     Kernel code
0x0000800000000000      User space (start)
0x00007ffffffde000      Stack
0x00007ffffffdc000      Guard page (prevents stack overflow)
0x00007ffff7ffa000      VDSO (virtual DSO - kernel code in user space)
0x00007ffff7dd9000      libc.so.6 (dynamic libraries)
0x00005555557dd000      Binary text/data
0x0000555555600000      Heap (grows up from here)
0x0000000000400000      Traditional text start (position-independent code uses lower addresses)
```

**3. Loading Segments**
- For each PT_LOAD segment in the program header:
  - Allocate virtual address range
  - Copy segment data from binary into virtual memory
  - Set permissions (r/w/x) based on segment flags
  - Register with memory management unit

**4. Dynamic Linking (for dynamically linked binaries)**
- Kernel loads the dynamic linker (interpreter): `/lib64/ld-linux-x86-64.so.2`
- Linker loads dependencies (libc, libpthread, user libraries)
- Performs symbol resolution and relocation
- Applies ASLR (address space layout randomization) if enabled
- Updates relocation entries in GOT (global offset table)

**5. Process Initialization**
- Kernel creates initial thread (main thread, PID = TID initially)
- Sets up argument vector (argv), environment (envp)
- Jumps to entry point: `_start()` (in crt1.o, provided by libc)
  - `_start()` calls global constructors, initializes libc
  - `_start()` calls `main(argc, argv, envp)`

**Reference**: CS:APP Section 7.5, "Linking"; Section 8.2, "Process Images"

### The Call Stack: The Machine's Function Invocation Mechanism

The stack is not abstract—it is a hardware-managed region of memory with explicit layout enforced by calling conventions. Understanding its structure at the machine code level is essential for diagnosing crashes, understanding performance, and writing performance-critical code.

**System V AMD64 ABI Calling Convention (Reference: AMD64 ABI Specification Section 3.2)**

This convention is used on Linux, BSD, and Unix variants. It specifies:

**Register Allocation**:
```
Purpose              Register  Preserved Across Call?
Return value (1st)   RAX       No (caller-owned)
Return value (2nd)   RDX       No (caller-owned)
Argument 1           RDI       No (caller-owned)
Argument 2           RSI       No (caller-owned)
Argument 3           RDX       No (caller-owned)
Argument 4           RCX       No (caller-owned)
Argument 5           R8        No (caller-owned)
Argument 6           R9        No (caller-owned)
Argument 7+          Stack     Caller pushes (right-to-left)
Callee-saved (1)     RBX       Yes (callee must preserve)
Callee-saved (2)     RBP       Yes (callee must preserve)
Callee-saved (3)     R12-R15   Yes (callee must preserve)
Caller-saved (1)     R10, R11  No (caller-owned)
Temporary            RAX, RCX, RDX, RSI, RDI, R8-R11  Caller-owned
```

**Function Prologue & Epilogue** (mandatory for ABI compliance):

When a function is called, the CPU executes `call`, which pushes the return address (RIP) onto the stack and transfers control:

```
Before:
RSP → [previous return address or caller's data]

After call instruction:
RSP → [return address] (pushed by CPU)
```

The called function must set up its stack frame:

**Prologue** (canonical form):
```asm
push %rbp              # Save caller's base pointer (line 1)
mov %rsp, %rbp         # Frame pointer = stack pointer (line 2)
sub $N, %rsp           # Allocate N bytes for local variables (line 3)
```

**Stack Frame Layout After Prologue**:
```
RSP + offset    Content
+0              [space for local variables]
+8              [saved RBP] ← RBP points here
+16             [return address from call instruction]
(caller's frame above)
```

**Epilogue** (reverses prologue):
```asm
mov %rbp, %rsp         # Deallocate locals
pop %rbp               # Restore caller's RBP
ret                    # Pop return address, jump to it
```

**The Red Zone** (Critical for Performance):

The 128-byte region directly below RSP is called the "red zone" and is reserved by the ABI. This memory is safe to use for temporary values WITHOUT allocating stack space (no `sub $N, %rsp`):

```
RSP - 0      [red zone boundary]
RSP - 128    [bottom of red zone]

Function can use RSP-1 through RSP-128 without sub $N, %rsp
Interrupt handlers must NOT use red zone (they overwrite it)
```

This is why leaf functions (functions that don't call others and don't use signals) often skip the prologue/epilogue entirely:

```asm
# Leaf function: no prologue needed
movq %rdi, %rax        # arg1 → rax
addq $10, %rax
ret                    # return in RAX

# This saves 4 instructions (2 push/pop, 2 mov) per call
```

**Example Call Sequence**:

```c
long add(long a, long b) {
    return a + b;
}

long caller() {
    long result = add(5, 10);
    return result * 2;
}
```

Assembly:
```asm
add:                    # Leaf function
    movq %rdi, %rax     # a → rax (argument 1)
    addq %rsi, %rax     # b added to rax
    ret                 # return

caller:
    push %rbp           # Prologue: save RBP
    mov %rsp, %rbp
    sub $16, %rsp       # 16 bytes local space (aligned to 16)

    movq $5, %rdi       # arg 1: RDI = 5
    movq $10, %rsi      # arg 2: RSI = 10
    call add            # CPU: push RIP, jump to add
                        # Stack: [RBP] [RIP of next instr] [locals]

    # RIP points here; RAX = return value (15)
    imulq $2, %rax      # RAX * 2

    mov %rbp, %rsp      # Epilogue
    pop %rbp
    ret
```

**Reference**: System V AMD64 ABI Release 1.0 (2018), Section 3.2 "Function Calling Sequence"

### Undefined Behavior and Compiler Optimizations

Undefined behavior (UB) is not merely a correctness issue—it enables compiler optimizations that can surprise you. Understanding what qualifies as UB and what optimizations it unlocks is critical for writing predictable, high-performance code.

**What the C Standard Says (C11 §3.4.3)**:
> Undefined behavior is behavior for which this International Standard imposes no requirements.

In practice, "no requirements" means the compiler can do anything, including:
- Assume the code never reaches that path
- Assume variables have certain values based on prior code
- Reorder code in unexpected ways

**Common Forms of UB Relevant to Systems Code**:

1. **Buffer overrun** (out-of-bounds access):
   ```c
   int arr[10];
   arr[100] = 5;  // UB: write to unowned memory
   ```
   The compiler may assume `arr[100]` is always uninitialized and optimize away reads.

2. **Signed integer overflow**:
   ```c
   int x = INT_MAX;
   int y = x + 1;  // UB: signed overflow
   ```
   The compiler is allowed to assume this never happens and remove checks:
   ```c
   if (x + 1 < x) { /* this path will be optimized away */ }
   ```

3. **Use-after-free**:
   ```c
   int *p = malloc(sizeof(int));
   free(p);
   *p = 5;  // UB: dereference freed memory
   ```

4. **Data race** (C11/C++11 onwards):
   ```c
   int x = 0;
   // Thread A:
   x = 1;
   // Thread B:
   int y = x;  // UB: concurrent access without synchronization
   ```

5. **Uninitialized variable read**:
   ```c
   int x;        // No initializer
   int y = x;    // UB: reading uninitialized variable
   ```

**Compiler Optimizations Enabled by UB Assumptions**:

**Example 1: Loop Optimization**
```c
void process_array(int *arr, int len) {
    for (int i = 0; i < len; i++) {
        printf("%d\n", arr[i]);  // May not actually call printf
    }
}
```

If the compiler cannot prove `arr` is valid (e.g., non-NULL, properly sized), it can assume UB and remove the entire loop. With `-fno-delete-null-pointer-checks`, this changes, but the compiler still applies aggressive optimizations.

**Example 2: Signed Integer Arithmetic**
```c
int is_positive(int x) {
    return x > 0;
}

int compare(int a, int b) {
    if (is_positive(a - b))  // UB: a - b may overflow
        return a > b;        // But compiler assumes a - b can't overflow
    return 0;
}
```

GCC optimizes this to:
```asm
cmp %esi, %edi  # a > b?
setg %al        # Set AL = 1 if greater
ret
```

The intermediate `a - b` is eliminated because signed overflow is UB.

**How to Write UB-Aware Code**:

1. **Use unsigned types for arithmetic that might overflow**:
   ```c
   unsigned long add_with_overflow_check(unsigned long a, unsigned long b) {
       if (a > ULONG_MAX - b) {
           // Overflow would occur (well-defined for unsigned)
           return 0;
       }
       return a + b;
   }
   ```

2. **Use `volatile` to prevent unsafe optimizations** (but this has performance cost):
   ```c
   volatile int x = 0;  // Compiler cannot assume UB won't occur
   x = x + 1;           // Will not be optimized away
   ```

3. **Use compiler flags to restrict optimizations**:
   - `-fwrapv`: Treat signed overflow as wrapping (not UB)
   - `-fno-strict-aliasing`: Allow type punning
   - `-fno-delete-null-pointer-checks`: Keep null checks

4. **Explicitly check bounds in hot paths**:
   ```c
   if (unlikely(ptr < valid_range_start || ptr > valid_range_end))
       return ERROR;  // Compiler cannot assume this path is never taken
   ```

**Reference**: CS:APP Chapter 2, "Representing and Manipulating Information"; Matz & Hubicka "System V AMD64 ABI"; Serebryany et al. "AddressSanitizer" (runtime UB detection)

### Volatile, Restrict, and Const: What They Actually Tell the Compiler

These three qualifiers are frequently misunderstood. They do not mean what casual programmers think they mean.

**`volatile`** (C99 §6.7.3)

Volatile is a promise to the compiler: "This variable may change outside the program's control; do not optimize away accesses."

The compiler must:
- Perform every read from a volatile variable (not cache it in a register)
- Perform every write to a volatile variable (not defer it)
- Maintain relative order of volatile accesses

The compiler may NOT:
- Eliminate volatile reads/writes
- Reorder volatile operations relative to each other
- Assume the variable has a specific value across operations

```c
volatile int timer_register = 0;  // Hardware memory-mapped register

while ((timer_register & 0x1) == 0) {
    // This loop must check timer_register each iteration
    // Without volatile, compiler assumes it never changes and optimizes to infinite loop
}
```

**Performance Impact**: volatile is expensive on performance-critical paths because it prevents optimizations. Use it only for memory-mapped hardware registers and rarely in application code.

**`restrict`** (C99 §6.7.3)

Restrict is a promise to the compiler: "No other pointer accesses the same memory during this function's execution."

```c
void copy_array(int * restrict dst, const int * restrict src, int len) {
    // Promise: dst and src do not overlap
    // Compiler can assume they are independent
    for (int i = 0; i < len; i++) {
        dst[i] = src[i];  // Can be vectorized, prefetched more aggressively
    }
}
```

Without restrict:
```asm
movl (%rsi,%rax,4), %ecx   # Load src[i]
movl %ecx, (%rdi,%rax,4)   # Store to dst[i]
                           # But what if dst overlaps src?
                           # Must reload and check after each store
```

With restrict:
```asm
movq (%rsi), %rax          # Load 2 elements from src
movq %rax, (%rdi)          # Store 2 elements to dst
```

The compiler can use vector instructions and prefetch more aggressively.

**`const`** (C99 §6.7.3)

Const means: "This variable will not be written."

```c
const int max_iterations = 1000;  // Cannot write to max_iterations
int const *ptr = &x;              // Cannot write *ptr (what ptr points to is const)
int * const ptr = &x;             // ptr itself is const (cannot reassign), but *ptr can be written
const int * const ptr = &x;       // Both ptr and *ptr are const
```

**Performance Impact**: const allows the compiler to:
- Place variable in read-only segment (.rodata instead of .data)
- Fold constant values at compile time
- Eliminate redundant loads (compiler knows value doesn't change)

```c
const int arr[] = {1, 2, 3, 4};
int sum = arr[0] + arr[1] + arr[2] + arr[3];

// Compiler optimizes to:
int sum = 10;  // Constant folding
```

**Combining const and restrict for maximum clarity**:
```c
void process(const int * restrict input, int * restrict output, int len) {
    // input: cannot be written, no aliasing
    // output: can be written, no aliasing with input
    // Compiler can apply aggressive vectorization and prefetching
    for (int i = 0; i < len; i++) {
        output[i] = input[i] * 2;
    }
}
```

**Reference**: C99 Standard Section 6.7.3; Intel Optimization Manual Section 12.4, "Restrict and Aliasing"

### Reading Compiler-Generated Assembly

To write performance-critical code, you must be able to read assembly and understand what the compiler generated. Tools and techniques:

**1. objdump: Disassemble Compiled Binaries**

```bash
gcc -O2 -g inference.c -o inference
objdump -d inference | grep -A 50 main:

# Output format:
# Address OpCode   Mnemonic  Operand, Operand
# 400400  <main>:
# 400400   55      push   %rbp
# 400401   48 89 e5   mov    %rsp,%rbp
```

Key flags:
- `-d`: Disassemble code sections only
- `-S`: Disassemble with original source interleaved (requires `-g`)
- `--prefix-addresses`: Show all addresses
- `-M intel`: Intel syntax (vs AT&T default)

**2. Godbolt (compiler-explorer.com)**

Online tool: compile C/C++ code and instantly see assembly. Select compiler (GCC, Clang, MSVC), optimization level (-O0 through -O3), and see side-by-side comparison.

Advantages:
- Instant feedback
- Can toggle optimizations
- See multiple compiler outputs
- Compare different languages (C vs Rust vs C++)

**3. perf: Sample Machine Code Execution**

```bash
perf record -g ./my_program    # Record call graph
perf report                     # View with annotation
perf report --stdio | head -50  # Textual output

# perf can show:
# - Which functions consume CPU time
# - Cache miss rates per instruction
# - Branch mispredictions
```

**4. Understanding Key Assembly Patterns**

**Memory operand addressing modes** (x86-64):
```asm
movq %rax, %rbx              # Register to register
movq $42, %rax               # Immediate to register
movq (%rsi), %rax            # Memory [RSI] to register (base addressing)
movq 8(%rsi), %rax           # Memory [RSI+8] to register (base + offset)
movq (%rsi,%rdi,4), %rax     # Memory [RSI + RDI*4] to register (base + index*scale)
movq 16(%rsi,%rdi,2), %rax   # Memory [RSI + RDI*2 + 16] (base + index*scale + offset)
```

The general addressing mode is: `offset(base, index, scale)`

**Example: Understanding a Loop**

```c
int sum_array(int *arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}
```

At -O2:
```asm
sum_array:
    xorl %eax, %eax         # sum = 0 (XOR is faster than MOV for 0)
    testl %esi, %esi        # len == 0?
    jle .L2                 # Jump if less-or-equal (loop doesn't execute)

    xorl %ecx, %ecx         # i = 0
.L3:                        # Loop label
    addl (%rdi,%rcx,4), %eax    # sum += arr[i]
                                # (%rdi,%rcx,4): [RDI + RCX*4]
    incl %ecx               # i++
    cmpl %esi, %ecx         # i < len?
    jl .L3                  # Jump if less
.L2:
    ret
```

**Key observations**:
- RDI holds arr (first argument)
- ESI holds len (second argument)
- EAX accumulates sum (also return value)
- ECX holds loop index i
- `(%rdi,%rcx,4)` accesses arr[i] (base + index*4 bytes per int)
- `incl %ecx` increments i
- `cmpl %esi, %ecx` compares i to len
- `jl .L3` branches back if i < len

**Reference**: CS:APP Chapter 3, "Machine-Level Representation of Programs"; Intel Software Developer's Manual Volume 1, Section 3.1, "Modes of Operation"; Agner Fog "Optimizing Subroutines in Assembly Language"

---

## 2. MENTAL MODEL

### The Execution Model as a State Machine

The following diagram represents the logical flow from source code through execution:

```
┌─────────────────────────────────────────────────────────────────┐
│                   SOURCE CODE (my_program.c)                    │
│         void process(int *arr, int len) { ... }                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ cc1 (compiler frontend)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INTERMEDIATE REPRESENTATION (IR)               │
│         (SSA form, optimizations applied: inlining,              │
│          dead code elimination, loop unrolling, etc.)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Code generator backend
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ASSEMBLY (my_program.s)                       │
│         process:                                                  │
│           movq %rdi, %rax                                        │
│           addq 8(%rbp), %rax  (relies on external symbols)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ as (assembler)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              OBJECT CODE (my_program.o)                          │
│         (machine code with unresolved external references)       │
│         Symbol table: process (offset 0x0, type=function)        │
│         Relocations: @plt entries for libc calls                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ ld (linker)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│          EXECUTABLE (./my_program, ELF binary)                   │
│         Program headers: PT_LOAD segments                        │
│         Entry point: _start (0x400400)                           │
│         Dynamic linker: /lib64/ld-linux-x86-64.so.2              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ execve() syscall, kernel loader
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│          VIRTUAL ADDRESS SPACE (process running)                 │
│                                                                   │
│  0xffffffffffffffff  │ Kernel space (inaccessible)              │
│  0x00007ffffffde000  │ Stack (grows down)                       │
│  0x00007ffff7ffa000  │ VDSO (virtual DSO)                       │
│  0x00007ffff7dd9000  │ libc.so.6 (0x200000 bytes)               │
│  0x00005555557dd000  │ Program .text, .data, .bss               │
│  0x0000555555600000  │ Heap (grows up, malloc/free)             │
│  0x0000000000000000  │ Unmapped                                  │
│                      │                                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │ CPU executes at entry point
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│          CPU EXECUTION (RIP points into code)                    │
│                                                                   │
│  RIP: 0x400400 (_start)                                          │
│    - Set up initial registers, stack                             │
│    - Call constructors (C++ global objects)                      │
│  RIP: 0x401000 (main)                                            │
│    - Prologue: push %rbp; mov %rsp, %rbp; sub $N, %rsp           │
│    - Function body: arithmetic, memory, branches                 │
│    - Epilogue: mov %rbp, %rsp; pop %rbp; ret                     │
│    - Return value in RAX                                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### The Call Stack: Detailed Layout

This diagram shows the exact state of the stack during a function call chain and the memory layout of each stack frame:

```
BEFORE: main() calls process_batch(arr, len)

        Higher addresses (top of stack, grows downward)

        0x7ffffffde000 ├─────────────────────────────┐
                       │ Stack space allocated but    │
                       │ not yet used                 │
        0x7ffffffdc000 ├─────────────────────────────┤◄─ RSP (before call)
                       │ [return addr to libc code]   │
                       │ [RBP of main]                │
                       │ [8 bytes of locals in main]  │
                       │ ← stack frame of main()
        0x7ffffffdc...  ├─────────────────────────────┤
                       │ [return addr within main]    │◄─ Pushed by 'call' instr
                       │ [new RBP value (RBP of main)]│◄─ Pushed by prologue
                       │ [local variables of process] │
                       │ [total: N bytes]             │◄─ RSP after prologue
                       │                              │
        0x0000000000000 └─────────────────────────────┘

                        Lower addresses (grows downward when decreasing RSP)

DETAILED STACK FRAME OF process_batch(arr, len):

After prologue: push %rbp; mov %rsp, %rbp; sub $32, %rsp

        (RBP)  0x7fff00000100 ├─────────────────────────────┐
                              │ [RBP of caller (main)]       │
        (RBP-8) 0x7fff000000f8 ├─────────────────────────────┤
                              │ [Return address (in main)]   │
        (RBP-16) 0x7fff000000f0 ├─────────────────────────────┤
                              │ [Local variable 1]           │
        (RBP-24) 0x7fff000000e8 ├─────────────────────────────┤
                              │ [Local variable 2]           │
        (RBP-32) 0x7fff000000e0 ├─────────────────────────────┤
                              │ [Local variable 3]           │
               RSP points here  (red zone below)             │
        (RSP-128) to (RSP)     │ 128-byte red zone           │
                              │ (safe to use without sub)    │
                              └─────────────────────────────┘

Red zone is critical for leaf functions (no prologue needed).
```

### ELF Loading and Dynamic Linking

```
DISK FILE (my_inference.o):                 RUNNING PROCESS (after execve):

ELF Header                                  Kernel allocates VAS
├─ Magic 0x7f454c46
├─ Entry point: 0x400400                   0x7fff0000 ┌─────────────┐
├─ Program header offset                    │  Stack   │
├─ Section header offset                    ├─────────┤
│                                           0x400000  │  VDSO       │
Program Headers (PT_LOAD):                 ├─────────┤
├─ Segment 1: .text (r-x)                  │  libc   │ (mmap'd)
│   File offset: 0x1000                    ├─────────┤
│   Virtual addr: 0x400000                 │  Program│
│   File size: 0x10000                     │  .text  │
│   Memory size: 0x10000                   │  .data  │
│                                           │  .bss   │ (BSS: zero-init)
├─ Segment 2: .data + .bss (rw-)            ├─────────┤
│   File offset: 0x11000                   │  Heap   │
│   Virtual addr: 0x410000                 │ (malloc)|
│   File size: 0x2000 (.data only)         ├─────────┤
│   Memory size: 0x4000 (.data + .bss)     0x000000  │ (unmapped)
│   (BSS occupies 0x2000 but not in file)  └─────────┘
│
├─ Segment 3: PT_DYNAMIC                   Dynamic Linking:
│   Lists dependencies, relocations        1. Kernel loads PT_INTERP:
│                                             /lib64/ld-linux-x86-64.so.2
├─ Segment 4: PT_INTERP                    2. Linker loads libc.so.6
│   /lib64/ld-linux-x86-64.so.2           3. Linker resolves symbols
│                                          4. Linker applies relocations
                                           5. Control passes to program entry
```

### Calling Convention: Register and Stack Layout

```
CALLER (before call instruction):

┌─ RAX     │ Caller can use freely (not preserved across call)
├─ RBX     │ Callee must preserve
├─ RCX     │ Caller can use freely
├─ RDX     │ Caller can use freely
├─ RSI     │ Caller can use freely
├─ RDI     │ Caller can use freely
├─ R8-R11  │ Caller can use freely
├─ R12-R15 │ Callee must preserve
├─ RSP     │ Stack pointer (points to return address after call)
└─ RBP     │ Frame pointer (optional, but conventional)

CALLER calls CALLEE with arguments:
  movq arg1_value, %rdi    # Argument 1 in RDI
  movq arg2_value, %rsi    # Argument 2 in RSI
  movq arg3_value, %rdx    # Argument 3 in RDX
  movq arg4_value, %rcx    # Argument 4 in RCX
  movq arg5_value, %r8     # Argument 5 in R8
  movq arg6_value, %r9     # Argument 6 in R9

  # For arg 7+, push to stack (right-to-left):
  movq arg8_value, %rax
  pushq %rax               # Argument 8 on stack
  movq arg7_value, %rax
  pushq %rax               # Argument 7 on stack

  call callee              # CPU: push RIP, jump to callee

CALLEE receives control:
  Stack now: [caller_RIP] [arg7] [arg8] ...

  PROLOGUE:
  pushq %rbp               # Save caller's RBP
  movq %rsp, %rbp          # RBP = RSP
  subq $local_size, %rsp   # Allocate space for locals

  Stack now:
  RBP   ├─ [caller_RBP]
  RBP-8 ├─ [caller_RIP]
  RBP-16├─ [local1]
  RBP-24├─ [local2]
  RSP   ├─ [free space]

  # Function body: access locals as -N(%rbp)
  movq -16(%rbp), %rax     # Load local1

  # Return value goes in RAX (or RDX:RAX for 128-bit)
  movq value, %rax

  EPILOGUE:
  movq %rbp, %rsp          # Deallocate locals
  popq %rbp                # Restore caller's RBP
  ret                      # CPU: pop RIP, jump to it

CALLER resumes after call:
  # Return value is in RAX
  movq %rax, result
```

---

## 3. PERFORMANCE LENS

### What This Means for Code Performance

**Principle 1: The Instruction Pipeline is Your Friend (and Enemy)**

Modern x86-64 CPUs can have instruction pipelines 20+ stages deep. Each instruction advances through stages:
- Fetch
- Decode
- Rename
- Execute
- Memory
- Writeback

When branches mispredictProfessional compilers, the pipeline flushes. All speculative work is discarded.

**Impact on Your Code**:
- Loops with unpredictable branching can lose 20+ cycles of speculation work per misprediction
- Simple linear code paths (no branches) allow full pipeline utilization
- Branch-heavy code (lots of if statements) can run at 1/4 the speed of branch-free code

**Example**: In an ML inference engine, if your per-token branching logic is complex:
```c
// BAD: Many branches
for (int i = 0; i < len; i++) {
    if (tokens[i] < vocab_size) {
        if (tokens[i] >= 0) {
            if (logits[i] > threshold) {
                // Process token
            }
        }
    }
}
```

Better: Eliminate branches with early exit or conditional load:
```c
// BETTER: Single branch per iteration
for (int i = 0; i < len; i++) {
    int token = tokens[i];
    if (token < 0 || token >= vocab_size)
        continue;
    float logit = logits[i];
    if (logit > threshold) {
        // Process
    }
}
```

Best: Use CMOV (conditional move, no branch):
```asm
cmp $0, %eax          # token < 0?
cmovl %edx, %eax      # if true, set token to sentinel
cmp $vocab_size, %eax # token < vocab_size?
cmovge %edx, %eax     # if false, set token to sentinel
# No mispredictions: CMOV never stalls
```

**Principle 2: Function Call Overhead is Non-Zero**

Every function call costs:
- 1 cycle to push return address (on stack)
- Several cycles to restore register state if caller-saved regs were overwritten
- Pipeline reset if the target is far away and branch prediction misses

For micro-kernels (batch size 1), this matters. For large batches, the amortized cost is negligible.

**Impact**:
- Inline small functions (< 5 instructions) to eliminate call overhead
- Use `__attribute__((always_inline))` when the compiler is unsure:
  ```c
  __attribute__((always_inline))
  static inline int clamp(int x, int min, int max) {
      return x < min ? min : (x > max ? max : x);
  }
  ```

**Principle 3: Stack Frame Overhead is Measurable**

The prologue (`push %rbp; mov %rsp, %rbp; sub $N, %rsp`) costs 3-4 cycles. The epilogue another 3 cycles.

For leaf functions (those that don't call other functions and don't need to preserve RBP), you can omit it:
```c
// Compiler with -fomit-frame-pointer will emit no prologue:
int add(int a, int b) {
    return a + b;  // No prologue/epilogue, just: mov %edi, %eax; add %esi, %eax; ret
}
```

**Impact**: Very important for tight loops that call small helper functions. Less important for main entry points.

**Principle 4: Red Zone Usage Eliminates Memory Writes**

The 128-byte red zone below RSP is safe to use without stack allocation. Leaf functions can use it to store temporary values without `sub $N, %rsp`:

```c
void leaf_fn(float *input, float *output) {
    // No prologue needed
    float temp1, temp2, temp3;  // Stored in red zone, not on stack
    // Compiler uses RBP-8, RBP-16, RBP-24 for these
    // No sub $N, %rsp instruction
}
```

**Impact**: Saves 1-2 cycles per function call by avoiding memory traffic.

**Principle 5: Calling Convention Register Pressure**

Only 6 integer registers are available for arguments. Beyond that, arguments go on the stack.

```c
// BAD: 7 arguments (last one on stack)
float complex_compute(float a, float b, float c, float d,
                     float e, float f, float g) {
    // g is accessed from stack; extra load instruction
}

// BETTER: Struct for logically grouped args
struct Params {
    float a, b, c, d, e, f, g;
};
float complex_compute(struct Params *p) {
    // All fields accessed via pointer dereferencing
    // Amortizes load cost
}
```

**Impact**: Accessing stack arguments adds 1-3 cycles (L1 cache hit).

**Principle 6: Volatile Prevents Optimization**

Using `volatile` on hot path variables disables:
- Register allocation (every access is a memory operation)
- Dead code elimination
- Common subexpression elimination

```c
volatile int counter = 0;
for (int i = 0; i < 1000; i++) {
    counter++;  // Must write to memory every iteration
}
```

At -O2, this compiles to:
```asm
xorl %eax, %eax
.L2:
    incl (%rdi)      # Atomic read-modify-write to memory
    incl %eax
    cmpl $1000, %eax
    jne .L2
```

Not volatile:
```asm
xorl %eax, %eax
.L2:
    incl %eax        # Stays in register
    cmpl $1000, %eax
    jne .L2
```

**Impact**: 10-100x slower with volatile on tight loops.

**Principle 7: Restrict and Aliasing Enable Vectorization**

```c
// Without restrict: compiler can't vectorize (pointer aliasing)
void add_vectors(int *dst, int *src, int len) {
    for (int i = 0; i < len; i++) {
        dst[i] += src[i];  // What if dst and src overlap?
    }
}

// Compiler generates scalar code:
movl (%rsi), %eax
addl %eax, (%rdi)
```

With restrict:
```c
void add_vectors(int * restrict dst, int * restrict src, int len) {
    // Compiler assumes no overlap
    for (int i = 0; i < len; i++) {
        dst[i] += src[i];
    }
}

// Compiler can vectorize:
movdqa (%rsi), %xmm0      # Load 4 ints from src
paddd (%rdi), %xmm0       # Add 4 ints from dst
movdqa %xmm0, (%rdi)      # Store 4 ints
```

**Impact**: 4-8x faster with restrict (vectorization enabled).

**Principle 8: Const Enables Constant Folding and Static Placement**

```c
const int max_tokens = 4096;

// Compiler replaces max_tokens with literal 4096 everywhere
if (num_tokens > max_tokens)
    return ERROR;
```

Const data lives in `.rodata` (read-only), which:
- Doesn't require runtime initialization
- Is often shared across process instances
- Avoids relocation records

Without const:
```c
int max_tokens = 4096;  // Lives in .data, requires runtime write
```

**Impact**: Saves initialization time and memory.

---

## 4. ANNOTATED CODE

### Example 1: Understanding the Call Stack with a Factorial Function

```c
// factorial.c: Illustrate calling convention and stack frames

#include <stdio.h>

// Leaf function: no prologue needed, uses 6 args in registers
// Arguments: RDI, RSI, RDX, RCX, R8, R9
// Return value: RAX
long leaf_add(long a, long b) {
    return a + b;  // No prologue: just add and return
}

// Non-leaf: calls printf, must follow ABI
long factorial(long n) {
    if (n <= 1) {
        printf("Base case: %ld\n", n);  // Calls printf, so prologue required
        return 1;
    }

    // Recursive case
    long prev = factorial(n - 1);  // Recursive call
    printf("Factorial(%ld) = %ld\n", n, prev * n);
    return prev * n;
}

int main() {
    long result = factorial(5);
    printf("Result: %ld\n", result);
    return 0;
}
```

**Compilation and Assembly**:

```bash
gcc -O0 -g factorial.c -o factorial  # No optimization, include debug info
objdump -d -M intel factorial | grep -A 50 "<factorial>:"
```

**Output (annotated)**:

```asm
0000000000400500 <factorial>:
400500:   55                      push   %rbp
         # Line 1: Save caller's RBP (prologue, required because we call printf)
         # RSP -= 8

400501:   48 89 e5                mov    %rsp,%rbp
         # Line 2: Frame pointer setup
         # RBP = RSP (RBP now points to saved RBP)

400504:   48 83 ec 18             sub    $0x18,%rsp
         # Line 3: Allocate 24 bytes (0x18) for local variables
         # RSP -= 24
         # Stack frame:
         #   RBP+0: [saved RBP]
         #   RBP-8: [return address] (pushed by call instruction)
         #   RBP-16: [local n] (stored for second printf call)
         #   RBP-24: [local prev] (return value from recursive call)

400508:   48 89 45 f0             mov    %rdi,-0x10(%rbp)
         # Line 4: Store argument n (RDI) to [RBP-16]
         # mov %rdi, -16(%rbp) — n = rdi (save arg)

40050c:   48 83 7d f0 01          cmp    $0x1,-0x10(%rbp)
         # Line 5: Compare n to 1
         # cmp $1, [RBP-16]

400511:   7f 12                   jg     400525 <factorial+0x25>
         # Line 6: Jump to .L2 (else branch) if n > 1
         # jg = jump if greater (unsigned)

400513:   48 8b 45 f0             mov    -0x10(%rbp),%rsi
         # Line 7: Load n from [RBP-16] → RSI (printf arg 2)

400517:   48 8d 3d 4a 00 00 00    lea    0x4a(%rip),%rdi
         # Line 8: Load format string address → RDI (printf arg 1)
         # Position-independent code: RIP-relative addressing

40051e:   b8 00 00 00 00          mov    $0x0,%eax
         # Line 9: RAX = 0 (number of vector arguments to printf, since no floats)

400523:   e8 28 ff ff ff          call   400450 <printf@plt>
         # Line 10: Call printf (through PLT)
         # CPU: push RIP (return address), jump to printf@plt

400528:   b8 01 00 00 00          mov    $0x1,%eax
         # Line 11: RAX = 1 (return value for base case)

40052d:   eb 30                   jmp    40055f <factorial+0x5f>
         # Line 12: Jump to epilogue

40052f:   48 8b 45 f0             mov    -0x10(%rbp),%rax
         # Line 13 (.L2): Load n → RAX

400533:   48 83 e8 01             sub    $0x1,%rax
         # Line 14: RAX = n - 1 (recursive argument)

400537:   48 89 c7                mov    %rax,%rdi
         # Line 15: RDI = RAX (argument to factorial)

40053a:   e8 c1 ff ff ff          call   400500 <factorial>
         # Line 16: Recursive call factorial(n-1)
         # CPU: push RIP, jump to factorial

40053f:   48 89 45 f8             mov    %rax,-0x8(%rbp)
         # Line 17: Store return value (RAX) to [RBP-8]
         # prev = rax

400543:   48 8b 55 f8             mov    -0x8(%rbp),%rdx
         # Line 18: Load prev → RDX (printf arg 3)

400547:   48 8b 45 f0             mov    -0x10(%rbp),%rax
         # Line 19: Load n → RAX

40054b:   48 0f af c2             imul   %rdx,%rax
         # Line 20: RAX = RAX * RDX = n * prev

40054f:   48 89 c2                mov    %rax,%rdx
         # Line 21: RDX = n * prev (for printf arg 3)

400552:   48 8b 45 f0             mov    -0x10(%rbp),%rax
         # Line 22: Load n → RAX (for printf arg 2)

400556:   48 89 c6                mov    %rax,%rsi
         # Line 23: RSI = n

400559:   48 8d 3d 1e 00 00 00    lea    0x1e(%rip),%rdi
         # Line 24: Load format string → RDI

400560:   b8 00 00 00 00          mov    $0x0,%eax
         # Line 25: RAX = 0 (no vector args)

400565:   e8 e6 fe ff ff          call   400450 <printf@plt>
         # Line 26: Call printf

40056a:   48 8b 45 f8             mov    -0x8(%rbp),%rax
         # Line 27: RAX = prev (for return)

40056e:   48 0f af 45 f0          imul   -0x10(%rbp),%rax
         # Line 28: RAX = prev * n (compute return value)

400573:   eb ec                   jmp    40055f <factorial+0x5f>
         # Line 29: Jump to epilogue

400575:   .L.epilogue:
400575:   c9                      leaveq
         # Line 30: Epilogue instruction
         # leaveq = mov %rbp, %rsp; pop %rbp
         # Deallocates locals, restores RBP

400576:   c3                      retq
         # Line 31: Return
         # CPU: pop RIP from stack, jump to it
```

**Key Observations**:

1. **Prologue** (lines 1-3):
   - `push %rbp` saves old frame pointer
   - `mov %rsp, %rbp` sets new frame pointer
   - `sub $0x18, %rsp` allocates 24 bytes

2. **Argument Passing**:
   - n is passed in RDI (first integer argument)
   - Saved to stack at [RBP-16] to preserve across recursive call

3. **Local Variables**:
   - Accessed relative to RBP: [RBP-8], [RBP-16], [RBP-24]
   - Compiler chose to spill n across the recursive call

4. **Recursive Call** (line 16):
   - Argument (n-1) loaded into RDI
   - `call` instruction pushes return address
   - After call, RAX holds result

5. **Epilogue** (lines 30-31):
   - `leaveq` = `mov %rbp, %rsp; pop %rbp`
   - `retq` pops return address and jumps

---

### Example 2: Demonstrating Undefined Behavior and Compiler Optimization

```c
// ub_demo.c: Show how UB enables aggressive optimizations

#include <string.h>

// Example 1: Signed overflow
int is_positive(int x) {
    return x > 0;
}

int compare_signed(int a, int b) {
    // UB: a - b can overflow (e.g., INT_MIN - 1)
    // Compiler assumes this UB never happens
    // Therefore, assumes a - b is representable
    // Therefore, assumes the subtraction can be reordered with comparison

    if (is_positive(a - b))
        return 1;  // a > b
    return 0;
}

// Example 2: Buffer overrun
void process_string(const char *src) {
    char dest[10];  // 10-byte buffer
    strcpy(dest, src);  // UB if src > 10 bytes (buffer overflow)
    // Compiler may assume src <= 10 bytes
    // Compiler may optimize away bounds checks
}

// Example 3: Uninitialized variable
int read_uninitialized(void) {
    int x;  // UB: never initialized
    return x + 0;  // UB: read uninitialized variable
    // Compiler may assume x has any value, or eliminate this function
}

// Example 4: Use-after-free and compiler optimizations
void use_after_free(void) {
    int *p = malloc(sizeof(int));
    *p = 42;
    free(p);
    int x = *p;  // UB: read after free
    // Compiler may:
    // (a) assume this never happens and remove it
    // (b) reorder *p = 42 to after the free (!)
    // (c) constant-fold x = 42 (since *p was just written)
}
```

**Compilation at -O2**:

```bash
gcc -O2 -S -masm=intel ub_demo.c
```

**Generated Assembly for `compare_signed`**:

```asm
compare_signed:
    mov %edi, %eax      # a → eax
    cmp %esi, %eax      # a > b? (compare without subtracting!)
    setg %al            # AL = 1 if greater, 0 otherwise
    movzx %eax, %eax    # Zero-extend AL to 32-bit
    ret
```

Notice: The compiler optimized away the subtraction entirely. It recognized that checking `a > b` is equivalent to checking whether `a - b > 0`, and for unsigned comparison, it can use the `cmp` instruction directly. This optimization would be invalid if the subtraction could overflow, but the compiler assumes UB never happens.

**Generated Assembly for `process_string`**:

With `-D_FORTIFY_SOURCE=0` (no runtime checks):

```asm
process_string:
    mov %rdi, %rsi      # src → rsi
    lea -10(%rsp), %rdi # dest → rdi (10-byte buffer)
    call strcpy@plt     # Call strcpy, assuming no overflow
    ret
```

The compiler generates a simple `strcpy` call. With `-D_FORTIFY_SOURCE=2`:

```asm
process_string:
    mov %rdi, %rsi
    lea -10(%rsp), %rdi
    mov $10, %rdx       # rdx = buffer size
    call __strcpy_chk@plt  # Call bounds-checked strcpy
    ret
```

---

### Example 3: Red Zone Usage in a Leaf Function

```c
// leaf_function.c: Demonstrating red zone usage

// Leaf function: no prologue needed, can use red zone
float dot_product_leaf(const float *a, const float *b, int len) {
    float sum = 0.0f;     // Stores in red zone, no stack allocation
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Non-leaf function: must allocate stack frame (calls printf)
void print_dot_product(const float *a, const float *b, int len) {
    float sum = dot_product_leaf(a, b, len);
    printf("Dot product: %f\n", sum);  // Calls printf
}
```

**Assembly (annotated)**:

```asm
# Leaf function: no prologue, no epilogue
dot_product_leaf:
    xorl %eax, %eax         # i = 0
    xorps %xmm0, %xmm0      # sum = 0.0 (XMM register for float)
    test %edx, %edx         # len == 0?
    jle .L2                 # Jump if done
.L3:                        # Loop:
    movss (%rdi,%rax,4), %xmm1   # Load a[i]
    mulss (%rsi,%rax,4), %xmm1   # Multiply by b[i]
    addss %xmm1, %xmm0      # Add to sum
    incl %eax               # i++
    cmp %edx, %eax
    jl .L3
.L2:
    ret                     # Return (no epilogue needed)

# sum accumulates in XMM0, never stored to memory
# No prologue, no epilogue: saves 6 instructions and memory traffic
```

**Non-leaf function (must allocate stack)**:

```asm
print_dot_product:
    push %rbp               # Prologue: save RBP
    mov %rsp, %rbp
    sub $8, %rsp            # Allocate 8 bytes for alignment
    movq %rdi, -8(%rbp)     # Save arguments (caller-saved)
    movq %rsi, -16(%rbp)
    movl %edx, -20(%rbp)

    # Call leaf_dot_product (no prologue in callee)
    call dot_product_leaf   # sum returned in XMM0

    movaps %xmm0, -24(%rbp) # Save return value
    # ... printf setup ...
    call printf

    mov %rbp, %rsp          # Epilogue
    pop %rbp
    ret
```

---

### Example 4: Restrict and Aliasing in Vector Operations

```c
// restrict_demo.c: Demonstrate restrict impact on vectorization

// Without restrict: compiler assumes aliasing, generates scalar code
void add_arrays_unsafe(int *dst, const int *src, int len) {
    for (int i = 0; i < len; i++) {
        dst[i] += src[i];  // What if dst and src overlap?
    }
}

// With restrict: compiler can vectorize
void add_arrays_safe(int * restrict dst, const int * restrict src, int len) {
    for (int i = 0; i < len; i++) {
        dst[i] += src[i];  // No aliasing possible
    }
}

// Combined with const and restrict: maximum optimization opportunity
void process_data(const int * restrict input,
                 int * restrict output,
                 int len) {
    // input: read-only, no aliasing
    // output: write-only, no aliasing with input
    for (int i = 0; i < len; i++) {
        output[i] = input[i] * 2 + 1;
    }
}
```

**Compilation with -O3 -march=native**:

```bash
gcc -O3 -march=native -S -masm=intel restrict_demo.c
```

**Generated Assembly (without restrict)**:

```asm
add_arrays_unsafe:
    test %edx, %edx         # len == 0?
    jle .L1
    xorl %eax, %eax         # i = 0
.L3:
    mov (%rsi,%rax,4), %ecx     # Load src[i]
    add %ecx, (%rdi,%rax,4)     # Add to dst[i]
                                # Note: Memory load-modify-write, not vectorized
    add $1, %eax
    cmp %edx, %eax
    jl .L3
.L1:
    ret
```

**Generated Assembly (with restrict)**:

```asm
add_arrays_safe:
    mov %edx, %eax
    sar $3, %eax            # Compute how many 256-bit vectors fit (8 ints per vector)
    test %eax, %eax
    je .L2

    xorl %ecx, %ecx         # Vector iteration counter
.L3:
    vmovdqa (%rsi,%rcx,4), %ymm0    # Load 8 ints from src
    vpaddd (%rdi,%rcx,4), %ymm0, %ymm0  # Add 8 ints from dst
    vmovdqa %ymm0, (%rdi,%rcx,4)    # Store 8 ints to dst
    add $8, %ecx
    cmp %eax, %ecx
    jl .L3
.L2:
    # Handle remainder
    ...
    ret
```

The restrict version:
- Uses 256-bit AVX vectors (8 integers per instruction)
- 8x fewer load/store instructions
- 8x faster than scalar version

---

## 5. EXPERT INSIGHT

### The Non-Obvious Truths That Separate Senior Engineers from Juniors

**Truth 1: The Calling Convention is Not Merely a Formality—It's a Contract**

Many programmers treat calling conventions as something the compiler handles automatically. In reality, understanding the calling convention is essential for:

- **Debugging crashes**: A stack corruption crash often stems from a mismatch between caller and callee stack frame expectations. If you know the calling convention by heart, you can immediately spot if a function prologue is wrong.

- **Hand-optimizing hot paths**: When profiling shows that a tight loop is bound by function call overhead, you can manually inline functions or use naked functions (no prologue/epilogue) for leaf functions.

- **Mixing languages**: If you're calling Rust code from C, or CUDA kernel launchers from C++, the calling convention must match. A casual mismatch will cause crashes that are terrifyingly difficult to debug.

**Truth 2: Stack Frames Have Overhead That Compoundsne**

A typical function prologue/epilogue costs 6-10 cycles:
- 1 cycle to push RBP
- 1 cycle for mov instruction
- 1 cycle for sub (if allocating locals)
- 2-3 cycles to restore

For a leaf function called 1 billion times in an inner loop, that's billions of wasted cycles. Modern compilers understand this and use `-fomit-frame-pointer` by default in optimization mode, but not all functions. The junior engineer assumes the compiler handles this perfectly. The senior engineer verifies with profiling that hot-path functions are indeed omitting frame pointers.

**Truth 3: Undefined Behavior Enables Compiler Optimizations That Can Surprise You**

A common mistake:

```c
// Code that "worked" in development but breaks in release
int *global_ptr;

void init() {
    global_ptr = malloc(sizeof(int) * 100);
}

int read_array(int i) {
    return global_ptr[i];  // What if i > 100?
}
```

In debug mode, this might work fine (bounds checking in malloc). But at -O3, the compiler might assume that bounds are always valid and reorder code:

```c
int read_array(int i) {
    // Compiler: "If we're here, i must be < 100 (no UB)"
    // So it optimizes the function to:
    int x = global_ptr[i];  // Direct access, no checks
    // If i == 101, x contains garbage, but the compiler doesn't care
}
```

The senior engineer always considers: "What UB is this code relying on?" and avoids writing code that depends on undefined behavior not happening.

**Truth 4: Virtual Address Space Layout is Not Symmetric**

Beginners often think of virtual memory as "all the memory is mapped at the same speed." In reality:

- Code and rodata live in the lowest addresses (usually 0x400000+)
- Libraries (libc, libpthread) live much higher
- Stack lives at the very top
- Heap grows up from somewhere in the middle

This has implications:
- Accessing rodata is often faster than accessing heap (fewer mispredictions if accessed sequentially)
- Stack variables are often faster than heap variables (better locality, cache warming)
- The ASLR (address space layout randomization) means the addresses change every run, but the relative layout stays the same

**Truth 5: The Red Zone is a Trap for Interrupt Handlers**

The 128-byte red zone is "safe to use without allocating stack space," but only for code that doesn't call other functions or handle interrupts.

A subtle bug:

```c
__attribute__((interrupt))  // Interrupt handler
void sig_handler(int sig) {
    int x;  // Uses red zone
    some_function();  // Calls another function
}
```

The interrupt handler uses the red zone, then calls `some_function()`, which also tries to use the red zone. If an interrupt fires during the `some_function()` call, both handlers fight over the same 128 bytes of memory. This is a classic source of non-reproducible crashes.

The senior engineer knows: interrupt handlers must not use the red zone, even if the function is a leaf.

**Truth 6: Restrict Doesn't Prevent Undefined Behavior—It Promises It Won't Happen**

```c
void add_vectors(int * restrict a, int * restrict b, int len) {
    for (int i = 0; i < len; i++) {
        a[i] += b[i];  // Compiler assumes a and b never overlap
    }
}

// Usage (INCORRECT):
int arr[10];
add_vectors(arr, arr, 10);  // Aliasing violation! Undefined behavior.
```

The compiler did NOT check for aliasing—it assumed the programmer promised no aliasing. If the programmer lied, the behavior is undefined. The compiler may generate code that crashes, silently corrupts data, or produces wrong results.

The junior engineer uses restrict carelessly. The senior engineer verifies at all call sites that the restrict contract is honored.

**Truth 7: const Does Not Prevent Writing—It Prevents YOU From Writing**

```c
const int x = 5;
x++;  // Compile error: cannot assign to const

// But:
int y = 10;
const int *p = &y;
*p = 20;  // Compile error: cannot dereference const pointer

// However:
int * const p2 = &y;
*p2 = 20;  // OK: p2 is const, but *p2 is not
```

const is a compiler construct. At runtime, a const variable can be overwritten if you bypass the compiler:

```c
const int locked = 1;
int *hack = (int *)&locked;
*hack = 0;  // Overwrites the "const" variable (UB, but happens)
```

The senior engineer uses const as a contract with the compiler, not as a runtime guarantee.

**Truth 8: ELF Relocation Types Determine Dynamic Linking Cost**

When a dynamically linked binary calls a library function, it doesn't jump directly to the function. Instead, it jumps through the PLT (Procedure Linkage Table) and GOT (Global Offset Table):

```asm
call printf@plt          # Not the real printf
```

The `printf@plt` entry is a stub that jumps to the real printf via the GOT. This adds 2-3 cycles of latency per call, plus potential cache misses if the PLT entry isn't warm.

PIE (Position-Independent Executable) code uses RIP-relative addressing for everything, which requires relocations at load time. A program with 10,000 relocations loads slower than one with 100 relocations.

The senior engineer understands how dynamic linking works and uses tools like `readelf -r` and `objdump -d` to inspect relocation records, then uses `LD_BIND_NOW` or link-time optimization to reduce this cost when necessary.

---

## 6. BENCHMARK / MEASUREMENT

### How to Measure This on Real Hardware

**Tool 1: objdump - Inspect the Binary**

```bash
# Disassemble entire binary:
objdump -d ./my_program | less

# Disassemble with source interleaved (requires -g):
objdump -S -d ./my_program | less

# Disassemble specific function:
objdump -d ./my_program | grep -A 30 "<main>:"

# Intel syntax (more readable):
objdump -M intel -d ./my_program | less

# Show relocation records:
objdump -r ./my_program | head -20
objdump -R ./my_program | head -20  # Dynamic relocations
```

**What to Look For**:
- Prologue/epilogue in all functions (not just leaves)
- Unnecessary `mov` instructions (register allocation issues)
- Branches in hot loops (branch prediction cost)
- Redundant loads/stores (spilling issues)
- Function inlining (or lack thereof)

**Tool 2: readelf - Inspect ELF Metadata**

```bash
# Show ELF header:
readelf -h ./my_program

# Show program headers (segments):
readelf -l ./my_program

# Show section headers:
readelf -S ./my_program

# Show symbol table:
readelf -s ./my_program | head -20

# Show relocations:
readelf -r ./my_program | head -20
readelf -Wr ./my_program | head -20  # Dynamic relocations

# Show dynamic symbols:
readelf -Ds ./my_program | head -20
```

**Key Metrics**:
- Number of PT_LOAD segments (should be 2-3 for a static binary, more for dynamic)
- Size of .text, .data, .bss (indicates code size and static allocations)
- Relocation count (indicates dynamic linking cost)
- Number of undefined symbols (indicates external dependencies)

**Example**:

```
$ readelf -h ./my_inference_engine
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                        0
  Type:                              EXEC (Executable file)
  Machine:                           Advanced Micro Devices X86-64
  Version:                           0x1
  Entry point address:               0x400e00
  Start of program headers:          64 (bytes into file)
  Start of section headers:          12456 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program header entry:      56 (bytes)
  Number of program headers:         9
  Size of section header entry:      64 (bytes)
  Number of section headers:         30
  Section header string table index: 29
```

The entry point is `0x400e00`. This is where the kernel jumps when executing the binary.

**Tool 3: nm - List Symbols**

```bash
# List all symbols:
nm ./my_program

# Sort by address:
nm -n ./my_program

# Show only function symbols:
nm -g ./my_program | grep " T "  # T = text (code) section

# Show undefined symbols (external references):
nm -u ./my_program

# Show symbol sizes:
nm -S ./my_program | head -20
```

**Example**:

```
$ nm -S ./my_inference_engine | head -20
0000000000400000 000000a0 T main
0000000000400100 000000f0 T inference_forward
0000000000400200 00000050 T activate_relu
                 U malloc
                 U free
                 U printf
```

The `U` symbols are undefined (external). The `T` symbols are defined in the text section. The size is the symbol size in bytes. A 0xf0 (240-byte) function is reasonably sized for inlining.

**Tool 4: ldd - Check Dynamic Dependencies**

```bash
# List dynamic dependencies:
ldd ./my_program

# Show address space layout:
ldd -v ./my_program  # Verbose: show all relocations

# Check if library is present:
ldd ./my_program 2>&1 | grep "not found"
```

**Example**:

```
$ ldd ./my_inference_engine
    linux-vdso.so.1 =>  (0x00007ffff7ffa000)
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007ffff7dd9000)
    /lib64/ld-linux-x86-64.so.2 => /lib64/ld-linux-x86-64.so.2 (0x00007ffff7ff8000)
```

The VDSO (virtual DSO) is a kernel-managed library that doesn't require file I/O.

**Tool 5: perf - Profile Execution and Sample Assembly**

```bash
# Record call graph:
perf record -g ./my_program
perf report

# Record with specific events:
perf record -e cycles,cache-misses,branch-misses ./my_program
perf report

# Annotate assembly with samples:
perf annotate

# Show hottest functions:
perf report --stdio | head -50

# Show function with assembly:
perf report --stdio | grep -A 50 "main"

# Record for 10 seconds:
perf record -F 99 ./my_program &  # -F: frequency (Hz)
sleep 10
kill %1
perf report
```

**Output Example**:

```
     99.85%  my_program  my_program           [.] inference_forward
                |
                --51.23%--inference_forward
                          |
                          |--30.21%--activate_relu
                          |          |
                          |          |--21.32%--sin@plt
                          |          |--08.54%--cosf@plt
                          |
                          |--12.43%--matrix_multiply
                          |
                          |--07.19%--softmax
```

This shows that `inference_forward` consumes 99.85% of the time, with `activate_relu` being the biggest subcomponent.

**Tool 6: time - Measure Execution Overhead**

```bash
# Measure wall clock time:
time ./my_program

# Measure CPU time only:
/usr/bin/time -f "%e elapsed %U user %S sys" ./my_program

# Measure with detailed stats:
/usr/bin/time -v ./my_program

# Repeated measurements:
for i in {1..5}; do time ./my_program; done
```

**Output**:

```
Command being timed: "./my_program"
User time (seconds): 4.23
System time (seconds): 0.15
Elapsed (wall clock) time (seconds): 4.39
Maximum resident set size (kbytes): 512000
Page reclaims: 123456
Page faults: 10
Involuntary context switches: 0
Voluntary context switches: 5
```

A small number of page faults and context switches is good. High numbers indicate contention or memory pressure.

**Tool 7: strace - Trace System Calls**

```bash
# Trace all syscalls:
strace ./my_program

# Count syscalls by type:
strace -c ./my_program

# Filter specific syscalls:
strace -e trace=open,close,read,write ./my_program

# Show syscall arguments:
strace -s 200 ./my_program  # -s: max string length to print
```

**What to Look For**:
- Too many syscalls in hot loops (expensive, should batch)
- Page faults (minor: reclaim from page cache; major: actual disk I/O)
- mmap calls (indicates dynamic allocation or library loading)
- Context switches (indicates contention, should be 0 for CPU-bound code)

**Tool 8: Godbolt Compiler Explorer**

Go to https://godbolt.org and:
1. Paste C code
2. Select compiler (GCC, Clang, MSVC)
3. Adjust optimization level (-O0, -O1, -O2, -O3)
4. See assembly immediately

**Useful for**:
- Quick feedback on compiler behavior
- Comparing multiple compiler versions
- Understanding specific compiler flags
- Inspecting specific functions in isolation

---

## 7. ML SYSTEMS RELEVANCE

### How This Directly Applies to ML Inference Engine Design

**Principle 1: Call Stack Overhead in Token-by-Token Inference**

A typical transformer inference engine processes tokens in a loop:

```c
for (int token_id = 0; token_id < max_tokens; token_id++) {
    logits = forward_pass(embeddings[token_id], kv_cache, layer_params);
    token_id = sample_next_token(logits);  // Calls random sampling
}
```

Each iteration may call dozens of functions (embedding lookup, matrix multiply, activation, layer norm, etc.). If these functions have heavy prologues/epilogues, the overhead compounds.

**Optimization Strategy**:
- Inline small functions (< 5 instructions) to avoid call overhead
- Use `-fomit-frame-pointer` to skip RBP saves in leaf functions
- Use `__attribute__((always_inline))` for hot-path functions:

```c
__attribute__((always_inline))
static inline float relu(float x) {
    return x > 0 ? x : 0;
}
```

For a batch size of 1 (interactive inference), call overhead can be 5-10% of total time. For batch size 128, it drops below 1%.

**Principle 2: Stack vs Heap Allocation for Intermediate Tensors**

When allocating intermediate activations during inference:

```c
// Option A: Stack allocation (fast, but limited size)
float activations[4096];  // Stack allocation
forward_layer(&activations[0], input, weights);

// Option B: Heap allocation (flexible, but slower)
float *activations = malloc(4096 * sizeof(float));
forward_layer(activations, input, weights);
```

Stack allocation is faster because:
- No malloc/free overhead
- Allocations are instant (just decrement RSP)
- Better cache locality (stack is warm)
- No fragmentation

However, stack is limited (~8 MB by default). For large models, you must use heap or pre-allocate a large buffer.

**Optimization Strategy for ML Engines**:
- Pre-allocate all intermediate buffers at startup (avoid malloc in hot path)
- Use stack for very small temporaries (< 1 KB)
- Use memory pools (pre-allocated heap chunks reused across iterations) for larger allocations

Example from an inference engine:

```c
struct InferenceContext {
    // Pre-allocated intermediate buffers
    float *layer_output;      // 4096 floats
    float *attention_scores;  // seq_len x seq_len
    float *softmax_out;       // seq_len
};

InferenceContext *ctx = alloc_inference_context(seq_len, hidden_dim);

for (int i = 0; i < num_tokens; i++) {
    forward_layer_inplace(ctx->layer_output, input, weights);  // Reuses same buffer
}
```

**Principle 3: Volatile and Restrict in Kernel Code**

When writing high-performance kernels, you need to guide the compiler:

```c
// Kernel: compute attention scores
// logits: output buffer (write-only)
// scores: input scores (read-only)
// len: sequence length
void compute_attention_scores(float * restrict logits,
                             const float * restrict scores,
                             const float * restrict mask,
                             int len) {
    for (int i = 0; i < len; i++) {
        logits[i] = scores[i] + mask[i];
    }
}
```

The `restrict` keywords tell the compiler:
- `logits` doesn't alias `scores` or `mask`
- The compiler can vectorize (load multiple values, process in parallel, store multiple values)
- No need to reload `scores` or `mask` after writing to `logits`

Without `restrict`, the compiler generates scalar code (1 operation per iteration). With `restrict`, it generates vectorized code (4-8 operations per iteration on AVX2).

**Impact**: 4-8x speedup on attention kernel.

**Principle 4: Function Inlining for Small Helper Functions**

ML inference code is full of small utility functions:

```c
static inline float square(float x) { return x * x; }
static inline float clamp(float x, float min, float max) {
    return x < min ? min : (x > max ? max : x);
}
static inline int index_2d(int row, int col, int cols) {
    return row * cols + col;
}
```

These should ALWAYS be inline. A function call overhead (8-10 cycles) is often larger than the function body (2-3 cycles).

**Compiler Behavior**:
- With `-O2` or `-O3`, the compiler usually inlines small functions automatically
- For uncertain cases, use `__attribute__((always_inline))`
- Never rely on LTO (link-time optimization) for this—it's not always enabled

**Principle 5: Calling Convention Impact on Batched Operations**

When processing a batch:

```c
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < seq_len; i++) {
        float x = batch_input[b][i];
        float y = apply_transform(x);  // Calls a function for each element
        batch_output[b][i] = y;
    }
}
```

With batch size 1: 1 call × seq_len = seq_len calls to `apply_transform`.
With batch size 128: 128 × seq_len calls.

The calling convention becomes a bottleneck. Options:

1. **Vectorize instead of calling**:
   ```c
   for (int b = 0; b < batch_size; b++) {
       for (int i = 0; i < seq_len; i += 4) {
           __m128 x = _mm_load_ps(&batch_input[b][i]);
           __m128 y = apply_transform_vec(x);
           _mm_store_ps(&batch_output[b][i], y);
       }
   }
   ```
   This reduces function calls by 4x.

2. **Fuse kernels** (eliminate function calls entirely):
   ```c
   // Instead of: embed → norm → attention → linear
   // Write: all_in_one_kernel() that does all four steps
   // Avoids inter-kernel function calls and memory traffic
   ```

**Principle 6: NUMA Awareness in Multi-Socket Inference**

On a 2-socket system, main memory is partitioned:
- Socket 0: 100 GB (fast local access, ~150 ns)
- Socket 1: 100 GB (slow remote access, ~300 ns)

The stack is thread-local and allocated on the socket where the thread is running. Heap allocations may be on the wrong socket.

**Optimization for ML**:
- Pin threads to specific sockets
- Allocate model weights and KV cache on the correct socket
- Use numactl to control allocation:
  ```bash
  numactl --membind=0 ./inference_engine  # All allocations on socket 0
  ```

**Principle 7: Inline Assembly for Critical Paths**

Sometimes, the compiler's generated code is not optimal. For critical paths (10%+ of total time), hand-optimization with inline assembly can help:

```c
// Matrix multiply inner loop: written in C first
float dot_product(const float *a, const float *b, int len) {
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Optimized with inline asm (hand-optimized for specific CPU):
float dot_product_asm(const float *a, const float *b, int len) {
    float sum;
    asm (
        "xorps %%xmm0, %%xmm0\n"  // sum = 0 in XMM0
        "movl %2, %%ecx\n"         // ecx = len
        "shrl $2, %%ecx\n"         // ecx /= 4 (process 4 floats per iteration)
        ".align 16\n"
        "1:\n"
        "movups (%0), %%xmm1\n"   // Load 4 floats from a
        "movups (%1), %%xmm2\n"   // Load 4 floats from b
        "mulps %%xmm2, %%xmm1\n"  // Multiply
        "addps %%xmm1, %%xmm0\n"  // Add to sum
        "addl $16, %0\n"           // a += 16 bytes
        "addl $16, %1\n"           // b += 16 bytes
        "decl %%ecx\n"
        "jnz 1b\n"
        // Horizontal add to get final sum
        : "=&r" (sum)
        : "r" (a), "r" (b), "r" (len)
        : "ecx", "xmm0", "xmm1", "xmm2"
    );
    return sum;
}
```

This hand-optimized version:
- Uses XMM registers (128-bit vectors)
- Processes 4 floats per iteration (vs compiler-generated scalar)
- Avoids prologue/epilogue (inline asm, no function overhead)
- Can be 4-8x faster than compiled C

---

## 8. PhD QUALIFIER QUESTIONS

1. **Stack Frame Layout and Undefined Behavior**:
   You have a C function with the following prologue:
   ```asm
   push %rbp
   mov %rsp, %rbp
   sub $16, %rsp
   ```
   An interrupt handler fires during execution of this function. The interrupt handler uses the red zone to store temporary values. Explain whether this is undefined behavior and why. What would happen if the interrupt handler calls a function that also uses the red zone?

2. **Calling Convention and Argument Passing**:
   Consider a function with 10 integer arguments. On x86-64 System V ABI, which arguments are passed in registers and which are on the stack? Draw the stack layout after the function prologue, showing where each argument is accessible. What is the performance implication of having arguments on the stack vs. in registers?

3. **ELF and Dynamic Linking**:
   An executable is compiled with `-fPIC -shared` (position-independent code, shared object). Explain the role of the GOT (Global Offset Table) and PLT (Procedure Linkage Table) in dynamic linking. What is the performance cost of a call to an external function compared to a static function? How does ASLR (address space layout randomization) interact with relocations?

4. **Undefined Behavior and Compiler Optimizations**:
   Consider this code:
   ```c
   int x = 0;
   for (int i = 0; i < n; i++) {
       x = x + i;  // Signed integer arithmetic
   }
   ```
   If `n` is large enough that `x` could overflow (undefined behavior in C), how might a compiler optimize this code differently than if `x` were unsigned? Show example assembly for both cases and explain the optimization.

5. **Restrict Qualifier and Alias Analysis**:
   Write a function that takes two pointer arguments with the restrict qualifier. Explain why restrict enables the compiler to generate better code (e.g., vectorization). What happens if you violate the restrict contract (call the function with aliasing arguments)? Is this undefined behavior?

---

## 9. READING LIST

**Essential Textbooks**:
- Bryant, Randal E., and David R. O'Hallaron. *Computer Systems: A Programmer's Perspective (3rd Edition)*. Pearson, 2015.
  - **Chapter 1**: "A Tour of Computer Systems" — overview of compilation and execution
  - **Chapter 3**: "Machine-Level Representation of Programs" — assembly, addressing modes, calling conventions
  - **Chapter 7**: "Linking" — ELF format, linking, dynamic linking, symbol resolution
  - **Chapter 8**: "Exceptional Control Flow" — process management, signal handlers, stack frames
  - **Chapter 9**: "Virtual Memory" — address space layout, paging, memory mapping

**Formal References**:
- AMD Inc. *System V Application Binary Interface AMD64 Architecture Processor Supplement (Version 1.0)* (2018)
  - **Section 3.1**: "Machine Types and Flags"
  - **Section 3.2**: "Function Calling Sequence" — register allocation, argument passing, return values, stack frame layout

- Intel Corporation. *Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 1: Basic Architecture*.
  - **Chapter 3**: "Instruction Set Reference, A-L" — instruction encoding, addressing modes
  - **Appendix C**: "Optimization Tips" — branch prediction, prefetching, register allocation

**Optimization Guides**:
- Agner Fog. *Optimizing Subroutines in Assembly Language: An Optimization Guide for x86 Platforms* (2022).
  - Comprehensive guide to x86-64 instruction timing, pipeline behavior, optimization techniques
  - Covers calling conventions, function prologue/epilogue, red zone

- Intel Corporation. *Intel 64 and IA-32 Architectures Optimization Reference Manual*.
  - Section 3.2: "Instruction Types and Encodings"
  - Section 4: "Performance Tuning"

**Tools and Analysis**:
- Serebryany, Konstantin, et al. "AddressSanitizer: A Fast Address Sanity Checker." USENIX ATC (2012).
  - Runtime tool for detecting use-after-free, buffer overruns, and other memory errors
  - Shows how UB is detected and reported

**Additional Resources**:
- Linux man pages: `man 2 execve`, `man 2 mmap`, `man 5 elf`
- GCC manual: https://gcc.gnu.org/onlinedocs/gcc/
- Clang/LLVM documentation: https://llvm.org/docs/
- Godbolt Compiler Explorer: https://godbolt.org/ (online assembly viewer and compiler)
