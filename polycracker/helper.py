#!/opt/conda/envs/pCRACKER_p27/bin/python
# All rights reserved.

def compute_correlation(mat):
    rowShape, columnShape = np.shape(mat)
    rowCombos = permutations(range(rowShape),rowShape)
    columnCombos = permutations(range(columnShape),columnShape)
    print mat
    maxR = []
    for idx,combos in enumerate([columnCombos,rowCombos]):
        for combo in combos:
            if idx == 0:
                matNew = mat[:, combo]
            else:
                matNew = mat[combo, :]
            coords = []
            for i in range(rowShape):
                for j in range(columnShape):
                    if matNew[i,j] > 0:
                        for k in range(matNew[i,j]):
                            coords.append((i,j))
            xy = np.array(coords)
            maxR.append(abs(pearsonr(xy[:,0],xy[:,1])[0]))
    return max(maxR)


def stats(arr):
    return (np.mean(arr), np.std(arr), np.min(arr), np.max(arr))


def label_new_windows(work_dir, windows_bed, original_subgenomes_bed):
    windows_bed = BedTool(windows_bed)
    scaffolds_subgenome_bed = BedTool(original_subgenomes_bed)
    labelled_bed = windows_bed.intersect(scaffolds_subgenome_bed,wa=True,wb=True).sort().merge(d=-1,c=7,o='distinct')
    ambiguous_bed = windows_bed.intersect(scaffolds_subgenome_bed,wa=True,v=True)
    bed_lines = []
    for line in str(ambiguous_bed).splitlines():
        if line:
            bed_lines.append(line.split()+['ambiguous'])
    for line in str(labelled_bed).splitlines():
        if line:
            if ',' in line:
                bed_lines.append(line.split()[:-1]+['ambiguous'])
            else:
                bed_lines.append(line.split())
    a = BedTool(bed_lines).sort()
    a.saveas(work_dir+'relabelled_windows.bed')
    new_labels = np.array([line.split()[-1] for line in str(a).splitlines()])
    pickle.dump(new_labels,open(work_dir+'new_labels.p','wb'))


