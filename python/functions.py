### Generic imports 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

### Plot setup
plt.rcParams.update({
    'figure.dpi': 200,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.max_open_warning": False,
})

def fiducial_cuts(df):
    df = df[df.g < 20.2] # reduces streaking 
    df = df[(0.5 <= df['b-r']) & (df['b-r'] <= 1)] # cold stellar streams in particular
    df = df[(np.abs(df['μ_ϕcosλ']) > 2) | (np.abs(df['μ_λ']) > 2)] # exclude stars near 0 proper motion
    return df

def make_plots(df, save_folder = "../plots"): 
    fig = plt.figure(figsize=(13,8), dpi=200, tight_layout=True)

    cmap = 'Greys'
    bins_0 = (np.linspace(-15,15,100), np.linspace(-15,15,100))
    bins_1 = (np.linspace(-20,10,100), np.linspace(-20,10,100))
    bins_2 = (np.linspace(0,3,100),np.linspace(9,20.2,100))

    ax = fig.add_subplot(231)
    h = ax.hist2d(df['ϕ'], df['λ'], cmap=cmap, cmin=1, vmax=250, bins=bins_0)
    ax.set_xlabel(r'$\phi~[^\circ]$',fontsize=20)
    ax.set_ylabel(r'$\lambda~[^\circ]$',fontsize=20)
    ax.set_xlim(-15,15);
    ax.set_ylim(-15,15);
    fig.colorbar(h[3], ax=ax)

    ax = fig.add_subplot(232)
    h = ax.hist2d(df['μ_ϕcosλ'], df['μ_λ'], cmap=cmap, cmin=1, bins=bins_1)
    ax.set_xlim(-20,10)
    ax.set_ylim(-20,10)
    ax.set_xlabel(r'$\mu_\phi^*$ [mas/yr]',fontsize=20)
    ax.set_ylabel(r'$\mu_\lambda$ [mas/yr]',fontsize=20)
    fig.colorbar(h[3], ax=ax)
    ax.set_title('Full Patch', fontsize=25, pad=15)

    ax = fig.add_subplot(233)
    h = ax.hist2d(df['b-r'], df['g'], cmap=cmap, cmin=1, bins=bins_2)
    ax.set_xlabel(r'$b-r$',fontsize=20)
    ax.set_ylabel(r'$g$',fontsize=20)
    ax.set_xlim(0,3)
    ax.invert_yaxis()
    fig.colorbar(h[3], ax=ax)

    ax = fig.add_subplot(234)
    h = ax.hist2d(df[df.stream]['ϕ'], df[df.stream]['λ'], cmap='Reds', bins=bins_0, cmin=1)
    ax.set_xlabel(r'$\phi~[^\circ]$',fontsize=20)
    ax.set_ylabel(r'$\lambda~[^\circ]$',fontsize=20)
    ax.set_xlim(-15,15);
    ax.set_ylim(-15,15);
    fig.colorbar(h[3], ax=ax)

    ax = fig.add_subplot(235)
    h = ax.hist2d(df[df.stream]['μ_ϕcosλ'], df[df.stream]['μ_λ'], cmap='Reds',cmin=1, bins=bins_1)
    ax.set_xlim(-20,10)
    ax.set_ylim(-20,10)
    ax.set_xlabel(r'$\mu_\phi^*$ [mas/yr]',fontsize=20)
    ax.set_ylabel(r'$\mu_\lambda$ [mas/yr]',fontsize=20)
    fig.colorbar(h[3], ax=ax)
    ax.set_title('Labeled Stream Stars', fontsize=25, pad=15)

    ax = fig.add_subplot(236)
    h = ax.hist2d(df[df.stream]['b-r'], df[df.stream]['g'], cmap='Reds', cmin=1, bins=bins_2)
    ax.set_xlabel(r'$b-r$',fontsize=20)
    ax.set_ylabel(r'$g$',fontsize=20)
    ax.set_xlim(0,3)
    ax.set_ylim(9,20.2)
    ax.invert_yaxis()
    fig.colorbar(h[3], ax=ax);

    plt.savefig(os.path.join(save_folder, "coords.pdf"))
    

# class SignalSideband:
#     """
#     A class for processing and plotting signal-sideband data.
    
#     Attributes:
#     df: DataFrame containing the input data.
#     sr_factor: Signal region factor.
#     sb_factor: Sideband region factor.
#     scan_over_mu_phi: If True, variable for scanning will be 'μ_ϕcosλ'. Else, it'll be 'μ_λ'.
#     df_slice: Processed DataFrame after applying signal-sideband conditions.
#     var: Variable selected based on scan_over_mu_phi.
#     sb_min, sb_max: Min and max bounds for sideband region.
#     sr_min, sr_max: Min and max bounds for signal region.
#     verbose: Boolean to control verbose output.
#     """
    
#     def __init__(self, df, sr_factor=1, sb_factor=3, scan_over_mu_phi=False, verbose=True):
#         """
#         Initialize SignalSideband with provided parameters.
#         """
#         self.df = df.copy()
#         self.sr_factor = sr_factor
#         self.sb_factor = sb_factor
#         self.verbose = verbose
#         self.scan_over_mu_phi = scan_over_mu_phi
#         self.df_slice = None
#         self.var = None
#         self.sb_min = None
#         self.sb_max = None
#         self.sr_min = None
#         self.sr_max = None
        
#         if self.verbose:
#             print("SR factor:", sr_factor)
#             print("SB factor:", sb_factor)

#     def process_data(self):
#         """
#         Process the data to create a sliced DataFrame based on signal-sideband conditions.
#         """
#         self.var = "μ_ϕcosλ" if self.scan_over_mu_phi else "μ_λ"

#         if self.verbose:
#             print("Scanning over "+str(self.var))

#         self.sb_min = self.df[self.df.stream][self.var].median() - self.sb_factor * self.df[self.df.stream][self.var].std()
#         self.sb_max = self.df[self.df.stream][self.var].median() + self.sb_factor * self.df[self.df.stream][self.var].std()
#         self.sr_min = self.df[self.df.stream][self.var].median() - self.sr_factor * self.df[self.df.stream][self.var].std()
#         self.sr_max = self.df[self.df.stream][self.var].median() + self.sr_factor * self.df[self.df.stream][self.var].std()

#         if self.verbose:
#             print("Sideband region: [{:.1f},{:.1f}) & ({:.1f},{:.1f}]".format(self.sb_min, self.sr_min, self.sr_max, self.sb_max))
#             print("Signal region: [{:.1f},{:.1f}]".format(self.sr_min, self.sr_max))

#         self.df_slice = self.df[(self.df[self.var] >= self.sb_min) & (self.df[self.var] <= self.sb_max)]
#         self.df_slice['label'] = np.where(((self.df_slice[self.var] >= self.sr_min) & (self.df_slice[self.var] <= self.sr_max)), 1, 0)

#         # Verbose output for stream counts
#         if  self.verbose:
#             # Total counts
#             n_total_stream = self.df.stream.value_counts().get(True, 0)
#             n_total_bkg = self.df.stream.value_counts().get(False, 0)
#             print("Total counts: {:,} stream and {:,} bkg events.".format(n_total_stream, n_total_bkg))
            
#             if "stream" in self.df.keys():
#                 # Stream counts and fractions
#                 n_sig_stream_stars = self.df_slice[self.df_slice.label == 1].stream.value_counts().get(True, 0)
#                 n_sig_bkg_stars = self.df_slice[self.df_slice.label == 1].stream.value_counts().get(False, 0)
#                 n_sideband_stream_stars = self.df_slice[self.df_slice.label == 0].stream.value_counts().get(True, 0)
#                 n_sideband_bkg_stars = self.df_slice[self.df_slice.label == 0].stream.value_counts().get(False, 0)

#                 print("Signal region has {:,} stream and {:,} bkg events ({:.2f}%).".format(n_sig_stream_stars, n_sig_bkg_stars,100*n_sig_stream_stars/n_sig_bkg_stars if n_sig_bkg_stars != 0 else 0))
#                 print("Sideband region has {:,} stream and {:,} bkg events ({:.2f}%).".format(n_sideband_stream_stars, n_sideband_bkg_stars, 100*n_sideband_stream_stars/n_sideband_bkg_stars if n_sideband_bkg_stars != 0 else 0))
#                 print("f_sig = {:.1f}* f_sideband.".format(n_sig_stream_stars/n_sig_bkg_stars/(n_sideband_stream_stars/n_sideband_bkg_stars) if n_sig_bkg_stars != 0 and n_sideband_bkg_stars != 0 else 0))



#     def plot_sb_data(self, save_folder=None):
#         """
#         Plot signal-sideband data with color-coded regions.
        
#         Arguments:
#         save_folder: Optional, path to the folder where plots will be saved.
#         """
#         bins = np.linspace(self.sb_min - (self.sr_min - self.sb_min), 
#                            self.sb_max + (self.sb_max - self.sr_max), 50)

#         fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5), dpi=300, tight_layout=True)
        
        
#         ax = axs[0]
#         N, bins, patches = ax.hist(self.df[self.df.stream == False][self.var], edgecolor='None', bins=bins)
#         for i in range(len(patches)):
#             if bins[i] < self.sb_min or bins[i] > self.sb_max:
#                 patches[i].set_alpha(0.4)
#                 patches[i].set_fc('lightgray')
#             elif (self.sb_min <= bins[i] and bins[i] < self.sr_min) or (self.sr_max < bins[i] and bins[i] <= self.sb_max):
#                 patches[i].set_fc('lightgray')
#             elif self.sr_min <= bins[i] and bins[i] <= self.sr_max:
#                 patches[i].set_fc('gray')

#         from matplotlib.patches import Patch
#         custom_legend = [Patch(facecolor='lightgray', alpha=0.25, label='Outer Region'),
#                          Patch(facecolor='lightgray', alpha=1, label='Sideband Region'),
#                          Patch(facecolor='gray', alpha=1, label='Signal Region'),
#                         ]
#         ax.set_title('Background Stars', fontsize=23)
#         if self.var == "μ_ϕcosλ": 
#             ax.set_xlabel(r'$\mu_\phi^*$ [mas/year]', fontsize=20)
#         else:
#             ax.set_xlabel(r'$\mu_\lambda$ [mas/year]', fontsize=20)
#         ax.set_ylabel('Number of Stars', fontsize=20)
#         ax.legend(handles=custom_legend, loc="upper left", frameon=False)

#         ax = axs[1]
#         N, bins, patches = ax.hist(self.df[self.df.stream == True][self.var], edgecolor='None', bins=bins)
#         for i in range(len(patches)):
#             if bins[i] < self.sb_min or bins[i] > self.sb_max:
#                 patches[i].set_alpha(0.25)
#                 patches[i].set_fc('crimson')
#             elif (self.sb_min <= bins[i] and bins[i] < self.sr_min) or (self.sr_max < bins[i] and bins[i] <= self.sb_max):
#                 patches[i].set_alpha(0.4)
#                 patches[i].set_fc('crimson')
#             elif self.sr_min <= bins[i] and bins[i] <= self.sr_max:
#                 patches[i].set_fc('crimson')

#         custom_legend = [Patch(facecolor='crimson', alpha=0.25, label='Outer Region'),
#                          Patch(facecolor='crimson', alpha=0.4, label='Sideband Region'),
#                          Patch(facecolor='crimson', alpha=1, label='Signal Region'),
#                         ]
#         ax.set_title('Stream Stars', fontsize=23)
#         if self.var == "μ_ϕcosλ": 
#             ax.set_xlabel(r'$\mu_\phi^*$ [mas/year]', fontsize=20)
#         else:
#             ax.set_xlabel(r'$\mu_\lambda$ [mas/year]', fontsize=20)
#         ax.set_ylabel('Number of Stars', fontsize=20)
#         ax.legend(handles=custom_legend, frameon=False)
        
#         if save_folder is not None:
#             if self.var == "μ_ϕcosλ": 
#                 plt.savefig(os.path.join(save_folder, "mu_phi.pdf"))    
#             else: 
#                 plt.savefig(os.path.join(save_folder, "mu_lambda.pdf"))

#         plt.show()



class SignalSideband:
    """
    A class for processing and plotting signal-sideband data.
    
    Attributes:
    df: DataFrame containing the input data.
    sr_factor: Signal region factor.
    sb_factor: Sideband region factor.
    scan_over_mu_phi: If True, variable for scanning will be 'μ_ϕcosλ'. Else, it'll be 'μ_λ'.
    df_slice: Processed DataFrame after applying signal-sideband conditions.
    var: Variable selected based on scan_over_mu_phi.
    sb_min, sb_max: Min and max bounds for sideband region.
    sr_min, sr_max: Min and max bounds for signal region.
    verbose: Boolean to control verbose output.
    process: Boolean controling process in initiation.
    
    
    Example use:
    signal_sideband_processor = SignalSideband(df)
    signal_sideband_processor.process_data()
    signal_sideband_processor.plot_sb_data()
    """
    def __init__(self, df, scan_over_mu_phi=False, sb_factor=1.0, sr_factor=1.0, verbose=True,process=False):
        self.df = df
        self.scan_over_mu_phi = scan_over_mu_phi
        self.sb_factor = sb_factor
        self.sr_factor = sr_factor
        self.verbose = verbose
        
        if process:
            self.process_data()

    def process_data(self):
        """
        Process the data to create a sliced DataFrame based on signal-sideband conditions.
        """
        self.var = "μ_ϕcosλ" if self.scan_over_mu_phi else "μ_λ"
        
        if self.verbose:
            print("Scanning over " + str(self.var))

        filtered_df = self.df.loc[self.df.stream, self.var]
        median_val = filtered_df.median()
        std_val = filtered_df.std()

        self.sb_min = median_val - self.sb_factor * std_val
        self.sb_max = median_val + self.sb_factor * std_val
        self.sr_min = median_val - self.sr_factor * std_val
        self.sr_max = median_val + self.sr_factor * std_val


        # Create the df_slice
        self.df_slice = self.df[(self.df[self.var] >= self.sb_min) & (self.df[self.var] <= self.sb_max)].copy()

        # Create the 'label' column
        self.df_slice['label'] = np.where((self.df_slice[self.var] >= self.sr_min) & (self.df_slice[self.var] <= self.sr_max), 1, 0)

        # Create the sr and sb dataframes
        self.sr = self.df_slice[self.df_slice.label == 1]
        self.sb = self.df_slice[self.df_slice.label == 0]

        
        if self.verbose:
            print("Sideband region: [{:.1f},{:.1f}) & ({:.1f},{:.1f}]".format(self.sb_min, self.sr_min, self.sr_max, self.sb_max))
            print("Signal region: [{:.1f},{:.1f}]".format(self.sr_min, self.sr_max))
            print("Total counts: SR = {:,}, SB = {:,}".format(len(self.sr), len(self.sb)))
            
            # Verbose output for stream counts
            if "stream" in self.df.keys():
                n_sig_stream_stars = self.sr.stream.value_counts().get(True, 0)
                n_sig_bkg_stars = self.sr.stream.value_counts().get(False, 0)
                n_sideband_stream_stars = self.sb.stream.value_counts().get(True, 0)
                n_sideband_bkg_stars = self.sb.stream.value_counts().get(False, 0)

                print("Signal region has {:,} stream and {:,} bkg events ({:.2f}%).".format(n_sig_stream_stars, n_sig_bkg_stars, 100 * n_sig_stream_stars/n_sig_bkg_stars if n_sig_bkg_stars != 0 else 0))
                print("Sideband region has {:,} stream and {:,} bkg events ({:.2f}%).".format(n_sideband_stream_stars, n_sideband_bkg_stars, 100 * n_sideband_stream_stars/n_sideband_bkg_stars if n_sideband_bkg_stars != 0 else 0))
                print("f_sig = {:.1f}X f_sideband.".format(n_sig_stream_stars/n_sig_bkg_stars/(n_sideband_stream_stars/n_sideband_bkg_stars) if n_sig_bkg_stars != 0 and n_sideband_bkg_stars != 0 else 0))

    def set_factors(self, sb_factor, sr_factor, verbose=False, process=True):
        """
        Update signal and sideband regions parameters.
        """

        self.sb_factor = sb_factor
        self.sr_factor = sr_factor
        self.verbose=verbose
        if process:
            self.process_data()

    def compute_ratios(self):
        """
        Compute sample size ratios between signal and sideband regions.
        Returns f_{stream,signal}/f_{stream,sideband} and  total(signal)/total(sideband)
        """
        n_sig_stream_stars = self.sr.stream.value_counts().get(True, 0)
        n_sig_bkg_stars = self.sr.stream.value_counts().get(False, 0)
        n_sideband_stream_stars = self.sb.stream.value_counts().get(True, 0)
        n_sideband_bkg_stars = self.sb.stream.value_counts().get(False, 0)
       
        f_signal_ratio = n_sig_stream_stars/n_sig_bkg_stars/(n_sideband_stream_stars/n_sideband_bkg_stars) if n_sig_bkg_stars != 0 and n_sideband_bkg_stars != 0 else 0
        sr_sb_ratio = len(self.sr) / len(self.sb) if len(self.sb) != 0 else 0

        return f_signal_ratio, sr_sb_ratio
    
    def plot_sb_data(self, save_folder=None):
        """
        Plot signal-sideband data with color-coded regions.
        
        Arguments:
        save_folder: Optional, path to the folder where plots will be saved.
        """
        bins = np.linspace(self.sb_min - (self.sr_min - self.sb_min), 
                           self.sb_max + (self.sb_max - self.sr_max), 50)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5), dpi=300, tight_layout=True)
        
        
        ax = axs[0]
        N, bins, patches = ax.hist(self.df[self.df.stream == False][self.var], edgecolor='None', bins=bins)
        for i in range(len(patches)):
            if bins[i] < self.sb_min or bins[i] > self.sb_max:
                patches[i].set_alpha(0.4)
                patches[i].set_fc('lightgray')
            elif (self.sb_min <= bins[i] and bins[i] < self.sr_min) or (self.sr_max < bins[i] and bins[i] <= self.sb_max):
                patches[i].set_fc('lightgray')
            elif self.sr_min <= bins[i] and bins[i] <= self.sr_max:
                patches[i].set_fc('gray')

        from matplotlib.patches import Patch
        custom_legend = [Patch(facecolor='lightgray', alpha=0.25, label='Outer Region'),
                         Patch(facecolor='lightgray', alpha=1, label='Sideband Region'),
                         Patch(facecolor='gray', alpha=1, label='Signal Region'),
                        ]
        ax.set_title('Background Stars', fontsize=23)
        if self.var == "μ_ϕcosλ": 
            ax.set_xlabel(r'$\mu_\phi^*$ [mas/year]', fontsize=20)
        else:
            ax.set_xlabel(r'$\mu_\lambda$ [mas/year]', fontsize=20)
        ax.set_ylabel('Number of Stars', fontsize=20)
        ax.legend(handles=custom_legend, loc="upper left", frameon=False)

        ax = axs[1]
        N, bins, patches = ax.hist(self.df[self.df.stream == True][self.var], edgecolor='None', bins=bins)
        for i in range(len(patches)):
            if bins[i] < self.sb_min or bins[i] > self.sb_max:
                patches[i].set_alpha(0.25)
                patches[i].set_fc('crimson')
            elif (self.sb_min <= bins[i] and bins[i] < self.sr_min) or (self.sr_max < bins[i] and bins[i] <= self.sb_max):
                patches[i].set_alpha(0.4)
                patches[i].set_fc('crimson')
            elif self.sr_min <= bins[i] and bins[i] <= self.sr_max:
                patches[i].set_fc('crimson')

        custom_legend = [Patch(facecolor='crimson', alpha=0.25, label='Outer Region'),
                         Patch(facecolor='crimson', alpha=0.4, label='Sideband Region'),
                         Patch(facecolor='crimson', alpha=1, label='Signal Region'),
                        ]
        ax.set_title('Stream Stars', fontsize=23)
        if self.var == "μ_ϕcosλ": 
            ax.set_xlabel(r'$\mu_\phi^*$ [mas/year]', fontsize=20)
        else:
            ax.set_xlabel(r'$\mu_\lambda$ [mas/year]', fontsize=20)
        ax.set_ylabel('Number of Stars', fontsize=20)
        ax.legend(handles=custom_legend, frameon=False)
        
        if save_folder is not None:
            if self.var == "μ_ϕcosλ": 
                plt.savefig(os.path.join(save_folder, "mu_phi.pdf"))    
            else: 
                plt.savefig(os.path.join(save_folder, "mu_lambda.pdf"))

        plt.show()




    
# def signal_sideband(df, sr_factor = 1, sb_factor = 3, save_folder=None, sb_min=None, sb_max=None, sr_min=None, sr_max=None, verbose=True, scan_over_mu_phi=False):
    
#     print("SR factor:", sr_factor)
#     print("SB factor:", sb_factor)
    
#     if scan_over_mu_phi:
#         var = "μ_ϕcosλ"
#     else:
#         var = "μ_λ"
        
#     print("Scanning over "+str(var))
    
#     if sb_min is not None:
#         sb_min = sb_min
#         sb_max = sb_max
#         sr_min = sr_min
#         sr_max = sr_max    
#     else: 
#         sb_min = df[df.stream][var].median()-sb_factor*df[df.stream][var].std()
#         sb_max = df[df.stream][var].median()+sb_factor*df[df.stream][var].std()
#         sr_min = df[df.stream][var].median()-sr_factor*df[df.stream][var].std()
#         sr_max = df[df.stream][var].median()+sr_factor*df[df.stream][var].std()
        
#     if verbose:
#         print("Sideband region: [{:.1f},{:.1f}) & ({:.1f},{:.1f}]".format(sb_min, sr_min, sr_max, sb_max))
#         print("Signal region: [{:.1f},{:.1f}]".format(sr_min,sr_max))
    
#     df_slice = df[(df[var] >= sb_min) & (df[var] <= sb_max)]
#     df_slice['label'] = np.where(((df_slice[var] >= sr_min) & (df_slice[var] <= sr_max)), 1, 0)
    
#     sr = df_slice[df_slice.label == 1]
#     sb = df_slice[df_slice.label == 0]
#     if verbose: print("Total counts: SR = {:,}, SB = {:,}".format(len(sr), len(sb)))

#     outer_region = df[(df[var] < sb_min) | (df[var] > sb_max)]
#     sb = df[(df[var] >= sb_min) & (df[var] <= sb_max)]
#     sr = df[(df[var] >= sr_min) & (df[var] <= sr_max)]    
        
#     bins = np.linspace(sb_min - (sr_min - sb_min), sb_max + (sb_max - sr_max), 40)

#     ### Make the plot
#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5), dpi=300, tight_layout=True) 
#     bins = np.linspace(sb_min - (sr_min - sb_min), sb_max + (sb_max - sr_max), 50)

#     ax = axs[0]
#     N, bins, patches = ax.hist(df[df.stream == False][var], edgecolor='None', bins=bins)
#     for i in range(len(patches)):
#         if bins[i] < sb_min or bins[i] > sb_max:
#             patches[i].set_alpha(0.4)
#             patches[i].set_fc('lightgray')
#         elif (sb_min <= bins[i] and bins[i] < sr_min) or (sr_max < bins[i] and bins[i] <= sb_max):
#             patches[i].set_fc('lightgray')
#         elif sr_min <= bins[i] and bins[i] <= sr_max:
#             patches[i].set_fc('gray')

#     from matplotlib.patches import Patch
#     custom_legend = [Patch(facecolor='lightgray', alpha=0.25, label='Outer Region'),
#                      Patch(facecolor='lightgray', alpha=1, label='Sideband Region'),
#                      Patch(facecolor='gray', alpha=1, label='Signal Region'),
#                     ]
#     ax.set_title('Background Stars', fontsize=23)
#     if var == "μ_ϕcosλ": 
#         ax.set_xlabel(r'$\mu_\phi^*$ [mas/year]', fontsize=20)
#     else:
#         ax.set_xlabel(r'$\mu_\lambda$ [mas/year]', fontsize=20)
#     ax.set_ylabel('Number of Stars', fontsize=20)
#     ax.legend(handles=custom_legend, loc="upper left", frameon=False);

#     ax = axs[1]
#     N, bins, patches = ax.hist(df[df.stream == True][var], edgecolor='None', bins=bins)
#     for i in range(len(patches)):
#         if bins[i] < sb_min or bins[i] > sb_max:
#             patches[i].set_alpha(0.25)
#             patches[i].set_fc('crimson')
#         elif (sb_min <= bins[i] and bins[i] < sr_min) or (sr_max < bins[i] and bins[i] <= sb_max):
#             patches[i].set_alpha(0.4)
#             patches[i].set_fc('crimson')
#         elif sr_min <= bins[i] and bins[i] <= sr_max:
#             patches[i].set_fc('crimson')

#     from matplotlib.patches import Patch
#     custom_legend = [Patch(facecolor='crimson', alpha=0.25, label='Outer Region'),
#                      Patch(facecolor='crimson', alpha=0.4, label='Sideband Region'),
#                      Patch(facecolor='crimson', alpha=1, label='Signal Region'),
#                     ]
#     ax.set_title('Stream Stars', fontsize=23)
#     if var == "μ_ϕcosλ": 
#         ax.set_xlabel(r'$\mu_\phi^*$ [mas/year]', fontsize=20)
#     else:
#         ax.set_xlabel(r'$\mu_\lambda$ [mas/year]', fontsize=20)
#     ax.set_ylabel('Number of Stars', fontsize=20)
#     ax.legend(handles=custom_legend, frameon=False);

#     ax.set_title('Stream Stars', fontsize=23)
#     if var == "μ_ϕcosλ": 
#         ax.set_xlabel(r'$\mu_\phi^*$ [mas/year]', fontsize=20)
#     else:
#         ax.set_xlabel(r'$\mu_\lambda$ [mas/year]', fontsize=20)
#     ax.set_ylabel('Number of Stars', fontsize=20)
#     if save_folder is not None:
#         if var == "μ_ϕcosλ": 
#             plt.savefig(os.path.join(save_folder,"mu_phi.pdf"))    
#         else: 
#             plt.savefig(os.path.join(save_folder,"mu_lambda.pdf"))    
    
#     if "stream" in df.keys():
#         try: n_sig_stream_stars = sr.stream.value_counts()[True]
#         except: n_sig_stream_stars = 0
#         try: n_sideband_stream_stars = sb.stream.value_counts()[True]
#         except: n_sideband_stream_stars = 0
#         try: n_sig_bkg_stars = sr.stream.value_counts()[False]
#         except: n_sig_bkg_stars = 0
#         try: n_sideband_bkg_stars = sb.stream.value_counts()[False]
#         except: n_sideband_bkg_stars = 0
          
#         if verbose:
#             print("Signal region has {:,} stream and {:,} bkg events ({:.2f}%).".format(n_sig_stream_stars, n_sig_bkg_stars,100*n_sig_stream_stars/n_sig_bkg_stars))
#             print("Sideband region has {:,} stream and {:,} bkg events ({:.2f}%).".format(n_sideband_stream_stars, n_sideband_bkg_stars, 100*n_sideband_stream_stars/n_sideband_bkg_stars))
#             print("f_sig = {:.1f}X f_sideband.".format(n_sig_stream_stars/n_sig_bkg_stars/(n_sideband_stream_stars/n_sideband_bkg_stars)))
#     return df_slice

def plot_results(test, top_n = [50, 100], save_folder=None, verbose=True, show=True):
    if save_folder is not None: 
        os.makedirs(save_folder, exist_ok=True)
    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(8,3), constrained_layout=True)
    bins=np.linspace(0,1,50)
    ax = axs[0]
    ax.tick_params(labelsize=12)
    ax.hist(test[test.label == 1].nn_score, bins=bins, histtype='step', linewidth=2, color="dodgerblue", label="Signal Region")
    ax.hist(test[test.label == 0].nn_score, bins=bins, histtype='step', linewidth=2, color="orange", label="Sideband Region")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title("Test Set")
    ax.set_xlabel("NN Score", size=12)
    ax.set_ylabel("Events", size=12)

    if "stream" in test.keys():
        ax = axs[1]
        ax.tick_params(labelsize=12)
        ax.hist(test[test.stream == False].nn_score, bins=bins, histtype='step', linewidth=2, color="grey", label="Not Stream")
        ax.hist(test[test.stream == True].nn_score, 
                bins=bins, histtype='step', linewidth=2, color="crimson", label="Stream")
        ax.legend(fontsize=12)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_title("Test Set")
        ax.set_xlabel("NN Score", size=12)
        ax.set_ylabel("Events", size=12);
    if save_folder is not None: 
        plt.savefig(os.path.join(save_folder,"nn_scores.png"))
    if show: plt.show()
    plt.close()
    
    ### Plot purities
    if "stream" in test.keys():
        # Scan for optimal percentage
        cuts = np.linspace(0.01, 50, 100)
        efficiencies = []
        purities = []
        for x in cuts:
            top_stars = test[(test['nn_score'] >= test['nn_score'].quantile((100-x)/100))]
            if True in top_stars.stream.unique():
                n_perfect_matches = top_stars.stream.value_counts()[True]
                stream_stars_in_test_set = test[test.stream == True]
                efficiencies.append(100*n_perfect_matches/len(stream_stars_in_test_set))
                purities.append(n_perfect_matches/len(top_stars)*100)
            else:
                efficiencies.append(np.nan)
                purities.append(np.nan)

        ### Choose a cut to optimize purity
        if not np.isnan(purities).all():
#             if verbose: print("Maximum purity of {:.1f}% at {:.2f}%".format(np.nanmax(purities),cuts[np.nanargmax(purities)]))
            cut = cuts[np.nanargmax(purities)]
            plt.figure(dpi=150)
            plt.plot(cuts, purities, label="Signal Purity")
            plt.xlabel("Top \% Stars, ranked by NN score")
            plt.legend()    
            if save_folder is not None: 
                plt.savefig(os.path.join(save_folder,"purities.png"))
            if show: plt.show()
            plt.close()

    ### Plot highest-ranked stars
    for x in top_n: # top N stars
        top_stars = test.sort_values('nn_score',ascending=False)[:x]
        if "stream" in test.keys():
            stream_stars_in_test_set = test[test.stream == True]
            if True in top_stars.stream.unique(): 
                n_perfect_matches = top_stars.stream.value_counts()[True] 
            else: 
                n_perfect_matches = 0 
        
            if verbose and show: print("Top {} stars: Purity = {:.1f}% ".format(x,n_perfect_matches/len(top_stars)*100))

        plt.figure(figsize=(5,3), dpi=150, tight_layout=True) 
        plt.title('Top {} Stars'.format(x))
        if "stream" in test.keys():
            plt.scatter(stream_stars_in_test_set.α_wrapped - 360, stream_stars_in_test_set.δ, marker='.', 
                    color = "lightgray",
                    label='Stream')
            plt.scatter(top_stars.α_wrapped - 360, top_stars.δ, marker='.', 
                    color = "lightpink",
                    label="Top Stars\n(Purity = {:.0f}\%)".format(n_perfect_matches/len(top_stars)*100))
            if True in top_stars.stream.unique(): 
                plt.scatter(top_stars[top_stars.stream].α_wrapped - 360, top_stars[top_stars.stream].δ, marker='.', 
                        color = "crimson",
                        label='Matches')
        else:
            plt.scatter(top_stars.α_wrapped - 360, top_stars.δ, marker='.', 
                    color = "crimson",
                    label="Top Stars") 
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        plt.xlim(test.α_wrapped.min() - 360,test.α_wrapped.max()- 360)
        plt.ylim(test.δ.min(),test.δ.max())
        plt.xlabel(r"$\alpha$ [\textdegree]")
        plt.ylabel(r"$\delta$ [\textdegree]")
        if save_folder is not None: 
            plt.savefig(os.path.join(save_folder,"top_{}_stars_purity_{}.pdf".format(x, int(n_perfect_matches/len(top_stars)*100))))
        if show: plt.show()
        plt.close()
    
def angular_distance(angle1,angle2): # function from David's file via_machinae.py, needed for FilterGD1 function
    # inputs are np arrays of [ra,dec]
    deltara=np.minimum(np.minimum(np.abs(angle1[:,0]-angle2[:,0]+360),np.abs(angle1[:,0]-angle2[:,0])),\
                          np.abs(angle1[:,0]-angle2[:,0]-360))
    deltadec=np.abs(angle1[:,1]-angle2[:,1])
    return np.sqrt(deltara**2+deltadec**2)

def FilterGD1(stars, gd1_stars):
    gd1stars=np.zeros(len(stars))
    for x in tqdm(gd1_stars):
        ra=x[0]
        dec=x[1]
        pmra=x[2]
        pmdec=x[3]
        foundlist=angular_distance(np.dstack((stars[:,3],stars[:,2]))[0],np.array([[ra,dec]]))
        foundlist=np.sqrt(foundlist**2+(stars[:,0]-pmdec)**2+(stars[:,1]-pmra)**2)   
        foundlist=foundlist<.0001
        if len(np.argwhere(foundlist))>1:
            print(foundlist)
        if len(np.argwhere(foundlist))==1:
            gd1stars+=foundlist
    gd1stars=gd1stars.astype('bool')
    return gd1stars,stars[gd1stars]

def load_file(filename):
    column_names = ["μ_δ", "μ_α", "δ", "α", "b-r", "g", "ϕ", "λ", "μ_ϕcosλ", "μ_λ"]
    gd1_stars = np.load('../gaia_data/gd1/gd1_stars.npy')
    df = pd.DataFrame(np.load(filename), columns = column_names)

    ### Label stream stars 
    is_stream, stream = FilterGD1(np.array(df), gd1_stars)
    df["stream"] = is_stream

    ### Wrap around alpha
    df['α_wrapped'] = df['α'].apply(lambda x: x if x > 100 else x + 360)
    return df