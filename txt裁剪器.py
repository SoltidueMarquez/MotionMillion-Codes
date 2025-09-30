import argparse


def filter_lines(input_path: str, output_path: str, prefix: str) -> int:
    kept = 0
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()  # 移除首尾空白字符
            if line.startswith(prefix):
                # 提取前缀后面的部分
                suffix = line[len(prefix):]
                # 检查是否是6位数字且小于200
                if suffix.isdigit() and len(suffix) == 6:
                    number = int(suffix)
                    if number < 200:  # 000200 等于 200
                        fout.write(line + '\n')
                        kept += 1
    return kept


def main():
    parser = argparse.ArgumentParser(description='Keep only lines starting with a specific prefix and with number less than 000200 in a txt file.')
    parser.add_argument('input', nargs='?', help='Path to input .txt file')
    parser.add_argument('-o', '--output', required=False, help='Path to output .txt file (default: <input>.filtered.txt)')
    parser.add_argument('-p', '--prefix', default='MotionGV/folder0/', help='Prefix to keep (default: MotionGV/folder0/)')
    args = parser.parse_args()

    inp = args.input
    if not inp:
        inp = input('Enter path to input .txt file: ').strip().strip('"')
    output_path = args.output if args.output else f"{inp}.filtered.txt"
    kept = filter_lines(inp, output_path, args.prefix)
    print(f"Wrote {kept} lines to {output_path}")


if __name__ == '__main__':
    main()