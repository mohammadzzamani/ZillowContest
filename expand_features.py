from sklearn import linear_model
import csv

zillow_file = 'data/z_reduced.csv'
language_file = 'data/csvX0.csv'
output_file = 'data/zillow_lang_features3.csv'

fin_z = open(zillow_file, 'r')
fin_l = open(language_file, 'r')
fout_zl = open(output_file, 'w')

z_lines = fin_z.readlines()
l_lines = fin_l.readlines()

primary_id_index_zillow = 0
primary_id_index_language = 0


# z_dict = dict()
l_dict = dict()
for l in l_lines[1:]:
    l_dict[l[primary_id_index_language]] = l[primary_id_index_language+1:].split(',')


def mult(lang_feat, zillow_feat, lang_feat_type=float, zillow_feat_type=float):
    if zillow_feat_type is str:
        # return None
        return ''

    try:
        return float(lang_feat) * float(zillow_feat)
    except ValueError:
        # return None
        return ''


# csv_wr = csv.writer(fout_zl, delimiter=',')

# Combine features
# for z in z_lines:
#     temp = list()
#     for x in z.split(','):
#         temp.append(x)
#     temp[-1] = temp[-1][:-1]
#     for l in l_dict[z[primary_id_index_zillow]]:
#         for x in z.split(',')[1:]:  # slices the primary key
#             temp.append(mult(l, x))
            # # temp.append(mult(l, x))
    # # print(len(temp), temp)  # output to file
    # # csv_wr.writerows(temp)
    # s = ''
    # for t in temp:
    #     s = s + str(t) + ','
    # s = s[:-1] + '\n'
    # fout_zl.write(s)

for z in z_lines:
    temp = list()
    temp.append(str(z.split(',')[primary_id_index_zillow]))
    for x in z.split(',')[1:]:
        temp.append(str(x))
        for l in l_dict[z[primary_id_index_zillow]]:
            temp.append(str(mult(l, x)))
    s = ''
    for t in temp:
        s = s + str(t) + ','
    s = s[:-1] + '\n'
    fout_zl.write(s)
