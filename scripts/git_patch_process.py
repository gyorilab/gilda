"""This is a helper script to edit the git patch created by
changes in grounding_cell_designer.py on the COVID-19 Disease Map XMLs.
The edited patch can then be applied and committed to that repo."""
import sys


def get_blocks(fname):
    """Get blocks of the diff that can be independently filtered."""
    with open(fname, 'r') as fh:
        lines = iter(fh.readlines())
        parts = []
        line = next(lines)
        while True:
            if line.startswith('diff --git'):
                block = [line]
                for line in lines:
                    if line.startswith('@@'):
                        break
                    block.append(line)
                parts.append(block)
            if line.startswith('@@'):
                block = [line]
                for line in lines:
                    if line.startswith('@@') or line.startswith('diff --git'):
                        break
                    block.append(line)
                parts.append(block)
            if line.startswith('\\ No newline'):
                parts[-1].append(line)
                try:
                    line = next(lines)
                except StopIteration:
                    break
            if not lines:
                break
    return parts


def filter_blocks(blocks):
    """Filter out spurious diffs caused by XML deserialization/serialization."""
    new_blocks = []
    for block in blocks:
        if any(l.startswith('-<rdf:RDF') for l in block):
            continue
        if any(l.startswith('-<math') for l in block):
            continue
        if any(l.startswith('-<sbml') for l in block):
            continue
        if any(l.startswith('-<body') for l in block):
            continue
        if any('&apos;' in l for l in block):
            continue
        new_blocks.append(block)
    return new_blocks


def dump_blocks(blocks, fname):
    """Dump filtered diffs back into a patch file."""
    with open(fname, 'w') as fh:
        for block in blocks:
            for line in block:
                fh.write(line)


if __name__ == '__main__':
    # 1. run the grounding_cell_designer.py script
    # 2. in the C19DM repo run git diff --binary > patch.diff
    # 3. run this script on patch.diff
    # 4. git apply patch_edited.diff
    patch_path = sys.argv[1]
    blocks = get_blocks(patch_path)
    blocks = filter_blocks(blocks)
    dump_blocks(blocks, patch_path[:-5] + '_edited.diff')
