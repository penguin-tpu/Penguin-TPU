# Verilog/SystemVerilog Style Guide

This guide defines the default formatting and coding conventions for Verilog and
SystemVerilog in this repository. It is written for both human contributors and
agents working on RTL, FPGA integration code, and testbenches.

When a local codebase pattern already exists inside a file or generated vendor
artifact, preserve that pattern unless there is a good reason to refactor it.
For new handwritten RTL, follow this guide.

## Naming

- Use `UpperCamelCase` for module names.
- Name handwritten `.v` and `.sv` files in `UpperCamelCase` as well, matching
  the primary module name in the file, similar to Java class/file naming.
- Use `lower_snake_case` for declarations, instance names, ports, signals,
  variables, functions, and parameters.
- Use `ALLCAPS` for constants and macros.
- Active-low signals must end with `_n`.
- For non-top modules, use `clock` and `reset` as the standard clock/reset port
  names.
- Board-facing or tool-facing top-level ports may keep platform-specific names
  when required by constraints or integration.

## Indentation And Spacing

- Use spaces only. Never use tabs.
- Use 4 spaces per indentation level.
- Configure the editor so the Tab key inserts spaces.
- Keep maximum line length at 120 characters. Staying closer to 100 characters
  is preferred when practical.
- Use blank lines to separate logical sections.
- Use one space around assignment operators.
- Use one space between control keywords and `(`:

```systemverilog
if (valid_i)
for (int i = 0; i < DEPTH; i = i + 1)
while (busy_o)
```

- Do not put a space before the opening parenthesis in function calls or
  function definitions:

```systemverilog
foo(bar);
function automatic logic is_done(input logic [3:0] state);
```

## Comments And Section Structure

- Comment generously with `//` or `/* ... */`.
- Prefer short, readable comments over long inline commentary.
- Use blank lines to separate logical regions of code.
- Put a blank line between a section-header comment and the first preprocessor
  directive in that section.
- Indent preprocessor directives to match the surrounding scope level.

Top-level module sections should use this banner style:

```systemverilog
//------------------------------------------------------------------------------
//  Logic declarations
//------------------------------------------------------------------------------
```

There should always be a blank line between the banner and the first HDL line
in that section.

Optional sub-sections inside a section may use a single-line comment:

```systemverilog
// State registers
```

## Module Organization

Organize handwritten modules in this order:

1. Parameters and `localparam`s
2. Logic declarations, types, structs, and state
3. Debugging printouts or simulation-only checks
4. Core logic sections

Use Verilog-2001 style module declarations and port declarations.

## Block Formatting

- Single-line statements may omit `begin`/`end`.
- Multi-line statements must use `begin`/`end`.
- `begin` stays on the same line as the controlling statement.
- `end` goes on its own line.
- Put `else` on its own line.
- Always include an `else` branch, even when the fallback is a hold/default
  behavior.

Example:

```systemverilog
if (enable_i) begin
    data_o = data_i;
end
else begin
    data_o = '0;
end
```

## Preferred Combinational And Sequential Style

- Prefer `assign` for simple combinational logic instead of creating extra
  `always_comb` blocks.
- For sequential logic, separate combinational next-state logic from sequential
  register-update logic whenever practical.
- Keep the sequential block focused on register updates.
- Keep transition and selection logic in `always_comb`.

## FSM Style

Use a two-process FSM style:

- A sequential block updates the current state and any delayed or registered
  strobes.
- A combinational block computes next state from current state and inputs.
- Default to hold behavior at the top of the combinational block:
  `next_state = current_state;`
- Keep transitions only in the combinational block.
- Use explicit priority with `if` / `else if` / `else`.
- Keep one `case` item per state.
- Use a typed `enum logic [...]` for state encodings when writing
  SystemVerilog FSMs.
- Add a defensive `default` case.

## Synthesizable RTL Rules

These rules apply to synthesizable hardware RTL. Some items are acceptable in
testbenches but should not appear in real design logic.

### Avoid In Synthesizable Design

- Avoid `negedge`. Prefer positive-edge logic and a cleaner clock/reset scheme.
- Avoid `initial` blocks. They may map on some FPGAs but do not represent a
  portable ASIC-safe reset strategy.
- Avoid delay expressions such as `#(...)`.
- Avoid `fork` / `join`.
- Avoid using `integer`, `real`, or similar types to describe actual hardware.
  Limited compile-time or loop-index use is acceptable.
- Avoid `*`, `/`, `%`, and `**` unless the hardware cost is intentional and
  justified.

Prefer shift-based arithmetic when possible:

```systemverilog
value_next = value_reg << 2;
value_next = value_reg >> 1;
```

### Use Carefully

- `task` and `function` are acceptable when they contain only synthesizable
  logic and no timing controls.
- `$clog2` and `$bits` are acceptable for elaboration-time constants, widths,
  and static declarations, but should not drive dynamic runtime logic.

## Testbench Exceptions

The following are acceptable in testbench or simulation-only code when needed:

- `initial`
- `#` delays
- `fork` / `join`
- simulation printouts and fatal checks

Keep simulation-only behavior clearly separated from synthesizable RTL.

## Practical RTL Checklist

Before checking in handwritten RTL, verify:

- Module name is `UpperCamelCase`.
- Non-top modules use `clock` and `reset`.
- Signals and instances use `lower_snake_case`.
- Active-low signals end in `_n`.
- Indentation is 4 spaces, with no tabs.
- `if`, `for`, and `while` use a space before `(`.
- Function calls do not use a space before `(`.
- Simple combinational logic uses `assign` where appropriate.
- Sequential logic is cleanly separated from next-state logic when practical.
- No `initial`, `#` delays, `fork/join`, or accidental `negedge` logic exists
  in synthesizable blocks.
- Section ordering and section banners are consistent.

## Example Skeleton

```systemverilog
module ExampleBlock #(
    //------------------------------------------------------------------------------
    //  Parameters / localparams
    //------------------------------------------------------------------------------

    parameter DEPTH = 4
) (
    input  logic clock,
    input  logic reset,
    input  logic valid_i,
    output logic ready_o
);

    //------------------------------------------------------------------------------
    //  Logic declarations
    //------------------------------------------------------------------------------

    typedef enum logic [1:0] {
        ST_RESET = 2'd0,
        ST_IDLE  = 2'd1,
        ST_BUSY  = 2'd2,
        ST_ERROR = 2'd3
    } state_e;

    state_e current_state;
    state_e next_state;

    logic valid_d;

    //------------------------------------------------------------------------------
    //  Debugging printouts (non-synthesizable)
    //------------------------------------------------------------------------------

    `ifndef SYNTHESIS
    always_ff @(posedge clock) begin
        if (!reset && current_state == ST_ERROR)
            $display("entered ST_ERROR");
    end
    `endif

    //------------------------------------------------------------------------------
    //  Core logic: outputs
    //------------------------------------------------------------------------------

    assign ready_o = (current_state == ST_IDLE);

    //------------------------------------------------------------------------------
    //  Core logic: sequential state
    //------------------------------------------------------------------------------

    always_ff @(posedge clock) begin
        if (reset) begin
            current_state <= ST_RESET;
            valid_d <= 1'b0;
        end
        else begin
            current_state <= next_state;
            valid_d <= valid_i;
        end
    end

    //------------------------------------------------------------------------------
    //  Core logic: combinational next state
    //------------------------------------------------------------------------------

    always_comb begin
        next_state = current_state;

        unique case (current_state)
            ST_RESET: begin
                next_state = ST_IDLE;
            end

            ST_IDLE: begin
                if (valid_i)
                    next_state = ST_BUSY;
                else
                    next_state = ST_IDLE;
            end

            ST_BUSY: begin
                if (!valid_i)
                    next_state = ST_IDLE;
                else
                    next_state = ST_BUSY;
            end

            default: begin
                next_state = ST_ERROR;
            end
        endcase
    end

endmodule
```
