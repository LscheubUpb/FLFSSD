import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats
from scipy.stats import ks_2samp
from tokenize import Double
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import math
import matplotlib.pyplot as plt
import os

def calculate_MIA(forget_similarity, retain_similarity):
    sorted_forget_similarities = np.sort(forget_similarity)
    sorted_retain_similarities = np.sort(retain_similarity)
    # sorted_valid_similarities = np.sort(train_similarity)

    print("Calculating Kolmogorov-Smirnov")

    data1 = sorted_forget_similarities
    data2 = sorted_retain_similarities

    # Define the common range for the x-axis based on min/max of both datasets
    xmin = min(data1.min(), data2.min())
    xmax = max(data1.max(), data2.max())
    bins = np.linspace(xmin, xmax, 100)  # Fixed bin edges

    # Compute cumulative histograms
    hist1, _ = np.histogram(data1, bins=bins, density=True)
    hist2, _ = np.histogram(data2, bins=bins, density=True)

    # Compute the CDFs
    cdf1 = np.cumsum(hist1) / np.sum(hist1)
    cdf2 = np.cumsum(hist2) / np.sum(hist2)

    D_f_r = np.max(np.maximum(cdf1-cdf2,0))
    
    """This is only for plotting for the thesis"""
    """Distribution"""
    # plt.clf()
    # plt.hist(sorted_retain_similarities, bins=30, alpha=0.5, label=f"Retain Similarities", density=True)
    # # plt.hist(sorted_valid_similarities, bins=30, alpha=0.5, label='Test Similarities', density=True)
    # plt.hist(sorted_forget_similarities, bins=30, alpha=0.5, label=f"Forget Similarities", density=True)
    # plt.title('Smiliarity Distributions')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # # plt.yscale('log')
    # plt.legend()
    # plt.show()
    # plt.savefig(f".\\distributions\\{os.getenv('ident','what')}.png", dpi=300, bbox_inches='tight')
    """end Distribution"""

    """Commulative distributions plot"""
    # data1 = sorted_forget_similarities
    # data2 = sorted_retain_similarities

    # # Define the common range for the x-axis based on min/max of both datasets
    # xmin = min(data1.min(), data2.min())
    # xmax = max(data1.max(), data2.max())
    # bins = np.linspace(xmin, xmax, 100)  # Fixed bin edges

    # # Compute cumulative histograms
    # hist1, _ = np.histogram(data1, bins=bins, density=True)
    # hist2, _ = np.histogram(data2, bins=bins, density=True)

    # # Compute the CDFs
    # cdf1 = np.cumsum(hist1) / np.sum(hist1)
    # cdf2 = np.cumsum(hist2) / np.sum(hist2)

    # # Get the midpoint of each bin for plotting
    # bin_centers = (bins[1:] + bins[:-1]) / 2

    # # Find the maximum difference (Kolmogorov-Smirnov Statistic)
    # ks_stat = np.max(np.abs(cdf1 - cdf2))
    # ks_index = np.argmax(np.abs(cdf1 - cdf2))
    # ks_x = bin_centers[ks_index]

    # # Plot cumulative histograms
    # plt.plot(bin_centers, cdf1, label='Forget Similarity Distribution', color='green')
    # plt.plot(bin_centers, cdf2, label='Retain Similarity Distribution', color='blue')

    # # Plot the vertical line at the point of maximum difference
    # plt.axvline(x=ks_x, color='red', linestyle='--', label=f'D_e_r = {ks_stat:.2f}')

    # # Annotate the KS statistic
    # # plt.text(ks_x + 0.1, 0.5, f'KS Stat = {ks_stat:.2f}', rotation=90, color='red')

    # plt.title('Cumulative Distribution Functions Forget-Retain')
    # plt.xlabel('Value')
    # plt.ylabel('Cumulative Probability')
    # plt.legend()
    # plt.show()
    '''End Commulative distributions plot'''
    """End of Section: This is only for plotting for the thesis"""

    print(f"D_f_r: {D_f_r}")

    return D_f_r


def calculate_PIC(forget_similarity, retain_similarity, test_similarity):
    s = forget_similarity
    t = test_similarity
    r = retain_similarity

    # print(f"t: mean={np.mean(t)}, std={np.std(t)}, min={np.min(t)}, max={np.max(t)}")
    # print(f"r: mean={np.mean(r)}, std={np.std(r)}, min={np.min(r)}, max={np.max(r)}")
    # print(f"s: mean={np.mean(s)}, std={np.std(s)}, min={np.min(s)}, max={np.max(s)}")
    # print(f"t: {t[:20]}")
    # print(f"r: {r[:20]}")
    # print(f"s: {s[:20]}")

    R_s = gaussian_kde(r)
    T_s = gaussian_kde(t)

    R_s_s = R_s(s)
    T_s_s = T_s(s)

    L_p_print = [f"{T_s(s[i])} / ({T_s(s[i])} + {R_s(s[i])})" for i in range(0,10)]

    L_p = [T_s(s[i]) / (T_s(s[i]) + R_s(s[i])) for i in range(len(R_s_s))]

    p = np.average([x for x in L_p if not np.isnan(x)])

    print("T_s(s[i]) / (T_s(s[i]) + R_s(s[i]))")
    print(f"L_p debug {L_p_print}")
    print(f"L_p {L_p[:20]}")
    print(f"p {p}")

    return p


# def calculate_PIC(forget_similarity, retain_similarity, train_similarity):
#     s = forget_similarity
#     t = train_similarity
#     r = retain_similarity

#     if(np.var(forget_similarity) == 0 and np.var(train_similarity) == 0 and np.var(retain_similarity) == 0):
#         return 0

#     T_s = gaussian_kde(t, bw_method=0.2)
#     R_s = gaussian_kde(r, bw_method=0.2)

#     log_L_r_s = np.sum(np.log(R_s(s)))
#     log_L_t_s = np.sum(np.log(T_s(s)))

#     print(f"log_L_r_s {log_L_r_s}")
#     print(f"log_L_t_s {log_L_t_s}")

#     try:
#         p = (1+math.exp(log_L_r_s-log_L_t_s))**-1
#     except:
#         # This occurs when the exponent is too big, then it gives a overflow error, in this case it is safe to assume p is either 0 or 1
#         if(log_L_r_s-log_L_t_s < 0):
#             p = 1
#         else:
#             p = 0

#     print(p)

#     return p





# def calculate_PIC(forget_similarity, retain_similarity, train_similarity):
#     s = forget_similarity
#     t = train_similarity
#     r = retain_similarity

#     # print(f"t: mean={np.mean(t)}, std={np.std(t)}, min={np.min(t)}, max={np.max(t)}")
#     # print(f"r: mean={np.mean(r)}, std={np.std(r)}, min={np.min(r)}, max={np.max(r)}")
#     # print(f"s: mean={np.mean(s)}, std={np.std(s)}, min={np.min(s)}, max={np.max(s)}")

#     # import matplotlib.pyplot as plt
#     # plt.hist(t, bins=50, alpha=0.5, label='train_similarity')
#     # plt.hist(r, bins=50, alpha=0.5, label='retain_similarity')
#     # plt.hist(s, bins=50, alpha=0.5, label='forget_similarity')
#     # plt.legend()
#     # plt.show()

#     max = np.max([np.max(t),np.max(r),np.max(s)])
#     min = np.min([np.min(t),np.min(r),np.min(s)])
#     t = (t - np.min(t)) / (max - min)
#     r = (r - np.min(r)) / (max - min)
#     s = (s - np.min(s)) / (max - min)

#     # print(f"t: mean={np.mean(t)}, std={np.std(t)}, min={np.min(t)}, max={np.max(t)}")
#     # print(f"r: mean={np.mean(r)}, std={np.std(r)}, min={np.min(r)}, max={np.max(r)}")
#     # print(f"s: mean={np.mean(s)}, std={np.std(s)}, min={np.min(s)}, max={np.max(s)}")

#     # plt.hist(t, bins=50, alpha=0.5, label='train_similarity_normalized')
#     # plt.hist(r, bins=50, alpha=0.5, label='retain_similarity_normalized')
#     # plt.hist(s, bins=50, alpha=0.5, label='forget_similarity_normalized')
#     # plt.legend()
#     # plt.show()

#     # T_s = KernelDensity(kernel='epanechnikov', bandwidth=0.2).fit(t)
#     T_s = gaussian_kde(t, bw_method=0.1)
#     # R_s = KernelDensity(kernel='epanechnikov', bandwidth=0.2).fit(r)
#     R_s = gaussian_kde(r, bw_method=0.1)

#     print(f"R_s(s): {R_s(s)[:25]}")
#     print(f"T_s(s): {T_s(s)[:25]}")

#     L_r_s = np.prod(R_s(s))
#     L_t_s = np.prod(T_s(s))

#     print(f"L_r_s {L_r_s}")
#     print(f"L_t_s {L_t_s}")

#     p = L_t_s / (L_t_s + L_r_s)

#     return p