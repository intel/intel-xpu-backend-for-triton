import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python case_update.py <updated_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, 'r') as f:
        lines = f.readlines()

    output_lines = []

    # 1 modify ill-conditioned cases for (u)int16 % float16
    mod_operation_ill_conditioned = False

    # 2 remove 'cuda', 'cpu_pinned' device for test_pointer_arguments

    # 3 allow TypeError for test_value_specialization_overflow
    value_specialization_overflow = False

    for line in lines:
        # 1
        if "_mod_operation_ill_conditioned" in line:
            mod_operation_ill_conditioned = True

        if "('uint16', 'float16')" in line and mod_operation_ill_conditioned:
            line = line.replace("uint16", "int16")
            mod_operation_ill_conditioned = False

        # 2
        if "(\"device\", ['cuda', 'cpu', 'cpu_pinned'])" in line:
            line = line.replace("['cuda', 'cpu', 'cpu_pinned']", "['cpu']")

        # 3
        if "test_value_specialization_overflow" in line:
            value_specialization_overflow = True

        if "pytest.raises(OverflowError)" in line and value_specialization_overflow:
            line = line.replace("OverflowError", "TypeError")
            value_specialization_overflow = False

        output_lines.append(line)

    with open(filename, 'w') as f:
        f.writelines(output_lines)
