Search.setIndex({docnames:["_autosummary/pykanto.app","_autosummary/pykanto.app.data","_autosummary/pykanto.app.main","_autosummary/pykanto.dataset","_autosummary/pykanto.parameters","_autosummary/pykanto.plot","_autosummary/pykanto.signal","_autosummary/pykanto.signal.analysis","_autosummary/pykanto.signal.cluster","_autosummary/pykanto.signal.filter","_autosummary/pykanto.signal.segment","_autosummary/pykanto.signal.spectrogram","_autosummary/pykanto.utils","_autosummary/pykanto.utils.compute","_autosummary/pykanto.utils.custom","_autosummary/pykanto.utils.io","_autosummary/pykanto.utils.paths","_autosummary/pykanto.utils.slurm","_autosummary/pykanto.utils.slurm.launch","_autosummary/pykanto.utils.slurm.tester","_autosummary/pykanto.utils.types","contents/FAQs","contents/basic-workflow","contents/deep-learning","contents/feature-extraction","contents/hpc","contents/installation","contents/interactive-app","contents/kantodata-dataset","contents/project-setup","contents/segmenting-files","contents/segmenting-vocalisations","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["_autosummary/pykanto.app.rst","_autosummary/pykanto.app.data.rst","_autosummary/pykanto.app.main.rst","_autosummary/pykanto.dataset.rst","_autosummary/pykanto.parameters.rst","_autosummary/pykanto.plot.rst","_autosummary/pykanto.signal.rst","_autosummary/pykanto.signal.analysis.rst","_autosummary/pykanto.signal.cluster.rst","_autosummary/pykanto.signal.filter.rst","_autosummary/pykanto.signal.segment.rst","_autosummary/pykanto.signal.spectrogram.rst","_autosummary/pykanto.utils.rst","_autosummary/pykanto.utils.compute.rst","_autosummary/pykanto.utils.custom.rst","_autosummary/pykanto.utils.io.rst","_autosummary/pykanto.utils.paths.rst","_autosummary/pykanto.utils.slurm.rst","_autosummary/pykanto.utils.slurm.launch.rst","_autosummary/pykanto.utils.slurm.tester.rst","_autosummary/pykanto.utils.types.rst","contents/FAQs.md","contents/basic-workflow.ipynb","contents/deep-learning.md","contents/feature-extraction.ipynb","contents/hpc.md","contents/installation.md","contents/interactive-app.md","contents/kantodata-dataset.ipynb","contents/project-setup.md","contents/segmenting-files.ipynb","contents/segmenting-vocalisations.ipynb","index.md"],objects:{"pykanto.app":[[1,0,0,"-","data"],[2,0,0,"-","main"]],"pykanto.app.data":[[1,1,1,"","embeddable_image"],[1,1,1,"","load_app_data"],[1,1,1,"","prepare_datasource"],[1,1,1,"","prepare_datasource_parallel"]],"pykanto.app.main":[[2,1,1,"","build_legend"],[2,1,1,"","get_markers"],[2,1,1,"","parse_boolean"],[2,1,1,"","prepare_legend"],[2,1,1,"","set_range"],[2,1,1,"","update_feedback_text"]],"pykanto.dataset":[[3,2,1,"","KantoData"]],"pykanto.dataset.KantoData":[[3,3,1,"","__init__"],[3,3,1,"","cluster_ids"],[3,3,1,"","get_units"],[3,3,1,"","open_label_app"],[3,3,1,"","plot"],[3,3,1,"","plot_example"],[3,3,1,"","plot_summary"],[3,3,1,"","prepare_interactive_data"],[3,3,1,"","reload"],[3,3,1,"","sample_info"],[3,3,1,"","save_to_disk"],[3,3,1,"","segment_into_units"],[3,3,1,"","subset"],[3,3,1,"","to_csv"],[3,3,1,"","write_to_json"]],"pykanto.parameters":[[4,2,1,"","Parameters"]],"pykanto.parameters.Parameters":[[4,3,1,"","__init__"],[4,3,1,"","add"],[4,4,1,"","dB_delta"],[4,4,1,"","dereverb"],[4,4,1,"","fft_rate"],[4,4,1,"","fft_size"],[4,4,1,"","gauss_sigma"],[4,4,1,"","highcut"],[4,4,1,"","hop_length"],[4,4,1,"","hop_length_ms"],[4,4,1,"","lowcut"],[4,4,1,"","max_dB"],[4,4,1,"","max_unit_length"],[4,4,1,"","min_silence_length"],[4,4,1,"","min_unit_length"],[4,4,1,"","num_cpus"],[4,4,1,"","num_mel_bins"],[4,4,1,"","silence_threshold"],[4,4,1,"","song_level"],[4,4,1,"","sr"],[4,4,1,"","subset"],[4,4,1,"","top_dB"],[4,3,1,"","update"],[4,4,1,"","verbose"],[4,4,1,"","window_length"]],"pykanto.plot":[[5,1,1,"","build_plot_summary"],[5,1,1,"","melspectrogram"],[5,1,1,"","mspaced_mask"],[5,1,1,"","rand_jitter"],[5,1,1,"","segmentation"],[5,1,1,"","show_minmax_frequency"],[5,1,1,"","show_spec_centroid_bandwidth"],[5,1,1,"","sns_histoplot"]],"pykanto.signal":[[7,0,0,"-","analysis"],[8,0,0,"-","cluster"],[9,0,0,"-","filter"],[10,0,0,"-","segment"],[11,0,0,"-","spectrogram"]],"pykanto.signal.analysis":[[7,1,1,"","approximate_minmax_frequency"],[7,1,1,"","get_mean_sd_mfcc"],[7,1,1,"","get_peak_freqs"],[7,1,1,"","spec_centroid_bandwidth"]],"pykanto.signal.cluster":[[8,1,1,"","hdbscan_cluster"],[8,1,1,"","reduce_and_cluster"],[8,1,1,"","reduce_and_cluster_parallel"],[8,1,1,"","umap_reduce"]],"pykanto.signal.filter":[[9,1,1,"","dereverberate"],[9,1,1,"","dereverberate_jit"],[9,1,1,"","gaussian_blur"],[9,1,1,"","get_norm_spectral_envelope"],[9,1,1,"","hz_to_mel_lib"],[9,2,1,"","kernels"],[9,1,1,"","mel_to_hz"],[9,1,1,"","mels_to_hzs"],[9,1,1,"","norm"],[9,1,1,"","normalise"]],"pykanto.signal.filter.kernels":[[9,4,1,"","dilation_kern"],[9,4,1,"","erosion_kern"]],"pykanto.signal.segment":[[10,2,1,"","ReadWav"],[10,2,1,"","SegmentMetadata"],[10,1,1,"","drop_zero_len_units"],[10,1,1,"","find_units"],[10,1,1,"","get_segment_info"],[10,1,1,"","onsets_offsets"],[10,1,1,"","save_segments"],[10,1,1,"","segment_file"],[10,1,1,"","segment_files"],[10,1,1,"","segment_files_parallel"],[10,1,1,"","segment_is_valid"],[10,1,1,"","segment_song_into_units"],[10,1,1,"","segment_song_into_units_parallel"]],"pykanto.signal.segment.ReadWav":[[10,3,1,"","__init__"],[10,3,1,"","as_dict"],[10,3,1,"","get_metadata"],[10,3,1,"","get_wav"],[10,4,1,"","wav_dir"]],"pykanto.signal.segment.SegmentMetadata":[[10,3,1,"","__init__"],[10,4,1,"","all_metadata"],[10,3,1,"","as_dict"],[10,3,1,"","get_metadata"],[10,4,1,"","index"]],"pykanto.signal.spectrogram":[[11,1,1,"","crop_spectrogram"],[11,1,1,"","cut_or_pad_spectrogram"],[11,1,1,"","extract_windows"],[11,1,1,"","flatten_spectrograms"],[11,1,1,"","get_indv_units"],[11,1,1,"","get_indv_units_parallel"],[11,1,1,"","get_unit_spectrograms"],[11,1,1,"","get_vocalisation_units"],[11,1,1,"","pad_spectrogram"],[11,1,1,"","retrieve_spectrogram"],[11,1,1,"","save_melspectrogram"],[11,1,1,"","window"]],"pykanto.utils":[[13,0,0,"-","compute"],[14,0,0,"-","custom"],[15,0,0,"-","io"],[16,0,0,"-","paths"],[17,0,0,"-","slurm"],[20,0,0,"-","types"]],"pykanto.utils.compute":[[13,1,1,"","calc_chunks"],[13,1,1,"","dictlist_to_dict"],[13,1,1,"","flatten_list"],[13,1,1,"","get_chunks"],[13,1,1,"","print_dict"],[13,1,1,"","print_parallel_info"],[13,1,1,"","timing"],[13,1,1,"","to_iterator"],[13,1,1,"","with_pbar"]],"pykanto.utils.custom":[[14,1,1,"","chipper_units_to_json"],[14,1,1,"","open_gzip"],[14,1,1,"","parse_sonic_visualiser_xml"]],"pykanto.utils.io":[[15,2,1,"","NumpyEncoder"],[15,1,1,"","copy_xml_files"],[15,1,1,"","get_unit_spectrograms"],[15,1,1,"","load_dataset"],[15,1,1,"","make_tarfile"],[15,1,1,"","makedir"],[15,1,1,"","read_json"],[15,1,1,"","save_json"],[15,1,1,"","save_songs"],[15,1,1,"","save_subset"],[15,1,1,"","save_to_jsons"]],"pykanto.utils.io.NumpyEncoder":[[15,3,1,"","default"]],"pykanto.utils.paths":[[16,2,1,"","ProjDirs"],[16,1,1,"","change_data_loc"],[16,1,1,"","get_file_paths"],[16,1,1,"","get_wavs_w_annotation"],[16,1,1,"","link_project_data"],[16,1,1,"","pykanto_data"]],"pykanto.utils.paths.ProjDirs":[[16,4,1,"","DATA"],[16,4,1,"","DATASET"],[16,4,1,"","DATASET_ID"],[16,4,1,"","FIGURES"],[16,4,1,"","PROJECT"],[16,4,1,"","RAW_DATA"],[16,4,1,"","REPORTS"],[16,4,1,"","RESOURCES"],[16,4,1,"","SEGMENTED"],[16,4,1,"","SPECTROGRAMS"],[16,3,1,"","__init__"],[16,3,1,"","append"],[16,3,1,"","update_json_locs"]],"pykanto.utils.slurm":[[18,0,0,"-","launch"],[19,0,0,"-","tester"]],"pykanto.utils.slurm.launch":[[18,1,1,"","submit_job"]],"pykanto.utils.types":[[20,2,1,"","Annotation"],[20,2,1,"","AttrProto"],[20,2,1,"","AudioAnnotation"],[20,2,1,"","Chunkinfo"],[20,4,1,"","Chunkinfo_"],[20,2,1,"","Metadata"],[20,2,1,"","SegmentAnnotation"],[20,2,1,"","ValidDirs"],[20,1,1,"","f_exists"],[20,1,1,"","is_list_of_int"],[20,1,1,"","is_list_of_str"]],"pykanto.utils.types.Annotation":[[20,4,1,"","ID"],[20,3,1,"","__init__"],[20,4,1,"","annotation_file"],[20,4,1,"","durations"],[20,4,1,"","end_times"],[20,4,1,"","label"],[20,4,1,"","lower_freq"],[20,4,1,"","start_times"],[20,4,1,"","upper_freq"]],"pykanto.utils.types.AttrProto":[[20,3,1,"","__init__"]],"pykanto.utils.types.AudioAnnotation":[[20,3,1,"","__init__"],[20,4,1,"","bit_rate"],[20,4,1,"","length_s"],[20,4,1,"","sample_rate"],[20,4,1,"","source_wav"]],"pykanto.utils.types.Metadata":[[20,4,1,"","ID"],[20,3,1,"","__init__"],[20,4,1,"","annotation_file"],[20,4,1,"","end"],[20,4,1,"","label"],[20,4,1,"","lower_freq"],[20,4,1,"","max_amplitude"],[20,4,1,"","min_amplitude"],[20,4,1,"","source_wav"],[20,4,1,"","start"],[20,4,1,"","upper_freq"],[20,4,1,"","wav_file"]],"pykanto.utils.types.SegmentAnnotation":[[20,4,1,"","ID"],[20,3,1,"","__init__"],[20,4,1,"","annotation_file"],[20,4,1,"","durations"],[20,4,1,"","end_times"],[20,4,1,"","label"],[20,4,1,"","lower_freq"],[20,4,1,"","start_times"],[20,4,1,"","upper_freq"]],"pykanto.utils.types.ValidDirs":[[20,4,1,"","DATASET_ID"],[20,4,1,"","PROJECT"],[20,4,1,"","RAW_DATA"],[20,3,1,"","__init__"]],pykanto:[[0,0,0,"-","app"],[3,0,0,"-","dataset"],[4,0,0,"-","parameters"],[5,0,0,"-","plot"],[6,0,0,"-","signal"],[12,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute"},terms:{"0":[4,5,7,8,9,10,11,13,16,21,22,24,25,28,30,31],"00":[22,25,30],"000":25,"001":[4,5],"01":[10,30],"02":[22,24,30,31],"03":4,"04":[22,28,30],"0415_05":[22,28],"048":31,"05":[22,24,30],"07":22,"08":22,"09":22,"0986848072562358":[22,28],"09868480725623585":[22,28],"0_build":25,"0m":22,"1":[3,4,7,8,9,15,16,21,22,24,25,28,30,31],"10":[3,8,9,10,22,28,31],"100":[3,5,9,24,25,30],"1000":4,"10000":[4,24],"1010":14,"1024":[4,14],"10448979591836727":[22,28],"10448979591836738":[22,28],"11":[22,24,28,31],"11000":31,"1102947845804989":[22,28],"12":[3,22,28],"120":31,"1219047619047619":[22,28],"12190476190476196":22,"126":22,"1277097505668935":22,"128":[4,9,31],"132":25,"13351473922902496":22,"138855":22,"138856":22,"139250":[],"14":[22,24,28],"14512471655328796":22,"15":[8,22,28],"150":14,"1519":[22,24,31],"1529":[],"16":[22,25,28,30],"169":22,"17081026":15,"187":31,"188250":[],"188388":22,"189776":22,"19":[],"194375":[],"1983":24,"1d":11,"1m":[22,24,31],"2":[4,7,8,13,22,24,28,30,31],"20":[3,25,28],"200":[25,30],"2018":1,"2021":[18,22,28,30],"20210502_040000":30,"2022":18,"2023":[22,24,31],"204":31,"2048":[4,24,31],"207805":[],"207806":[],"21":[22,28,31],"22":[22,28],"22050":[4,9,10,11,24],"224":[4,31],"22518":21,"22m":[22,24,31],"23":[22,28,30],"234":[],"2392":[],"24":28,"240":24,"247aa5075e06337d":30,"24f319055fdf2205":22,"25":28,"2506":[],"26":28,"268":31,"27":22,"28":[22,24,31],"29024943310657597":22,"2d":[3,9,11],"2f":28,"2m":22,"2v":30,"3":[4,7,9,22,23,26,28],"30":[4,25],"300":25,"32":[24,28],"32000":31,"32m127":[22,24,31],"34829931972789113":22,"351275":22,"356706":22,"36":22,"369":24,"36m":22,"37":24,"376":31,"38":31,"38893424036281177":22,"39m":[22,24,31],"4":[4,23,24,30],"40000":25,"41":1,"44":21,"45":22,"45859410430839004":22,"46439909297052157":22,"472":25,"48":[25,30],"48000":[],"5":[2,4,8,9,22,28,30,31],"50":[3,5],"500":[1,31],"5108390022675737":22,"512":[11,24,31],"516":31,"520000":22,"527":24,"5322":[],"536":31,"54032744":13,"55":[],"556":25,"5694":22,"5739":22,"5863038548752835":22,"592108843537415":22,"5922":22,"5977":24,"6":[],"600000":22,"615328798185941":22,"63853":31,"64":1,"65":[4,24],"65db":4,"666701":22,"673711":22,"68":31,"6907936507936508":22,"712":31,"7314285714285714":22,"74":22,"768":30,"8":25,"80b1d3":3,"81":28,"8265":[],"8266":[22,24,31],"83":30,"835":22,"8359183673469388":22,"836":31,"847":22,"8491":24,"866667":22,"876":31,"8dd3c7":3,"9":[24,26,28],"90000":25,"92":30,"947":[],"95":7,"96":22,"976":22,"98":22,"99":28,"9984":[],"boolean":2,"break":13,"case":[3,26,29,31],"class":[3,4,9,10,13,15,16,20,21,22,27,30],"default":[1,2,3,4,5,7,8,9,10,11,13,14,15,16,22],"do":[21,22,24,25,27,29,30,31],"export":[15,23],"final":[11,24,31],"float":[1,3,4,5,7,8,9,10,11,13,14,20,24],"function":[1,2,3,5,7,8,9,10,11,13,14,15,16,18,20,22,24,25,27,29],"import":[3,16,22,23,24,25,29,30,31],"int":[1,2,3,4,5,7,8,9,10,11,13,14,20],"long":[22,29,32],"new":[3,4,10,15,16,26,27,28,30,32],"return":[1,2,3,4,5,7,8,9,10,11,13,14,15,16,24,30],"short":[24,25,30],"switch":26,"throw":21,"true":[1,3,5,7,10,11,13,14,15,16,22,24,26,27,29,30,31],"try":[3,15,21],"while":[30,31],A:[1,3,5,7,11,13,15,16,25,28],And:[26,30],As:[24,25],But:22,For:[10,15,22,25,27,28,29,30],If:[3,11,21,22,24,25,26,29,30,31],In:[3,10,23,24,31],Is:10,It:[16,26,27,29],NO:10,No:16,Not:4,ONE:20,Of:22,One:[3,5],Or:28,That:[9,24],The:[3,4,11,14,15,16,22,23,24,26,32],Then:[30,31],There:[22,23,25],These:[15,16,24,27],To:[22,25,26,27,30],_:15,__dict__:[13,30],__future__:24,__init__:[3,4,10,16,20,30],_description_:[1,10,14],_redis_password:25,_reduce_and_cluster_r:22,abl:22,about:[13,20,28],abov:3,acceler:8,access:[24,28,29,31],account:9,accuraci:3,acess:30,acoust:[25,32],activ:26,actual:29,adapt:[13,23],add:[3,4,5,25,27,30],add_to_dict:30,address:25,advis:26,after:[3,10,25,27,28,31],again:[21,22,24,30],agnost:30,agre:30,aim:24,algorithm:[22,25,26],alia:20,all:[1,3,4,5,8,10,11,22,29,30],all_metadata:[10,30],allow:[3,4,27],allow_nan:15,along:[3,5,10,26,30],alreadi:[3,16,20,22,24,28,29],also:[18,25,26,27],am:[16,30],among:[3,22,29],amplitud:[5,10,22,31],an:[3,5,7,9,10,11,13,14,15,16,20,21,24,25,27,28,29,30],analys:[24,30,32],analysi:24,ani:[3,4,5,10,13,14,15,23,24,25,26,27,28,29,30,31],anim:[3,32],annot:[5,10,14,16,20,24,30,31],annotation_fil:20,annotation_path:16,anoth:[22,31],anyth:4,app:[3,22,32],append:[14,15,16],appli:[10,11,28],applic:[0,1,25,26,27,30],approach:24,approxim:[5,7,8,9,25],approximate_minmax_frequ:[7,24],apt:21,ar:[3,7,8,10,16,18,20,22,24,25,27,28,29,30],arbitrari:15,arc:18,archiv:32,arg:20,argument:[3,5,10,15,18,25,29],argv:25,around:[13,24,30],arr:5,arrai:[5,7,8,9,10,11,24],articl:[23,31,32],artist:30,as_dict:[10,24],ascend:3,assign:[3,16,22,27],associ:28,assum:[16,29,30],attach:10,attr:[20,30],attribut:[3,10,13,16,20,30],attrproto:20,audio:[3,7,10,11,14,16,20,22,24,29,30,31],audio_format:30,audio_metadata:24,audio_sect:10,audioannot:[10,20],audiomoth:30,auto_class:[2,8,27],autom:30,automat:[3,4,22,27],automaticali:3,autoreload:21,avail:[4,10,21,23,30],averag:[1,3,4,8,11,22,25],ax:5,ax_percentag:2,axi:2,b32:[22,28],b3de69:3,b:[2,20,25],back:23,background:1,bad:31,badwidth:[5,24],bandpass:[9,11],bandwidth:7,bandwith:5,bar:[5,10,13],base:[1,3,9,10,15,22,23],bash:[18,25],basic:[7,27,28,32],batteri:30,bc80bd:3,bebada:3,becaus:28,been:[3,22,24,27,29],befor:[20,22,25],behaviour:18,behind:32,being:[21,24],belong:[3,11],below:[3,4,18,29,30],benefit:22,bengalese_finch:[16,30,31],better:24,between:[4,23],beyond:5,bigbird2020:25,bigbird:[15,16],bigbird_0:15,bigbird_2021:29,bigexternaldr:16,bin:[3,4,5,7,9,25],binari:5,bird:[3,22,23,32],birdsong:31,bit:25,bit_depth:[22,30],bit_rat:20,bitrat:30,blur:9,bokeh:[2,3],bone:5,bool:[1,2,3,4,5,7,8,10,11,13,14,15,16],both:[3,25],bound:[3,5,11],bout:[20,22],box:[3,5,11,18,30],breath:24,breath_not:24,broken:[15,16],browser:[3,22,27],build:[2,3,13,25,29],build_legend:2,build_plot_summari:5,burrow:24,burst:24,bw:24,c:[18,24,25,26],calc_chunk:13,calcul:[4,5,7,13,25,28],call:[15,18,21,24,25,27,29,30,31],callabl:10,can:[3,5,7,10,16,18,21,22,23,24,25,26,27,28,29,30,31],cannot:22,captur:24,carri:25,categori:3,category20_20:3,caus:[3,16],ccebc5:3,cd:26,center:5,centr:11,centroid:[5,7,24],cepstral:7,chain:13,challeng:26,chanc:18,chang:[3,16,18,30],change_data_loc:16,channel:30,characteris:9,check:[3,10,16,20,22,25,27,31],check_circular:15,chipper:14,chipper_units_to_json:[14,24],choos:[3,24],chosen:28,chunk:[10,13,20,22,24,29],chunk_length:13,chunkinfo:[13,20],chunkinfo_:20,chunksiz:[13,20],clariti:29,classifi:27,clean:[26,32],clip:9,clone:26,cluster:[3,22],cluster_id:[3,22,27,28],cluster_resourc:25,cmap:5,code:[1,2,9,10,15,18,22,23,29],coeffici:7,coincid:16,collaps:11,collect:[5,11,13],colour:[2,3,5],colour_bar:5,column:[2,8,15,22,27,28],columndatasourc:[1,2],com:[13,15,26],combin:[10,20,24],come:[3,5,24],command:25,comment:30,common:[3,13,26,29,31],compat:21,complet:23,composit:15,compress:22,comput:[4,10,11,12,18,32],computation:25,concis:30,conda:26,condit:31,conserv:8,consid:[8,10,25],consist:22,consolid:10,construct:16,consum:30,contain:[3,5,7,10,11,13,14,16,22,24,26,27,28,30,32],content:[13,28],contribut:26,control:[8,16,29],conveni:[8,10,23,29],convert:9,coordin:[3,8],copi:[3,15],copy_xml_fil:15,core:[2,25],correct:27,correspod:15,correspond:3,could:[15,21,30],count:11,coupl:24,cover:20,cpu:[1,3,4,10,21,25],crash:21,creat:[3,9,10,11,15,16,22,24,25,26,27,28,29,30,31],criteria:10,crop:[5,11],crop_i:11,crop_spectrogram:11,crop_x:11,csv:[3,28],cuda:26,cuml:[8,26],current:[3,14,16],custom:[13,20,24,27],customannot:30,cut:11,cut_or_pad_spectrogram:11,d9d9d9:3,d:30,dashboard:[22,24,31],data:[2,3,5,7,8,9,10,11,15,16,20,22,23,24,27,28,30,31],data_dir:[10,29,30],data_path:[24,31],databas:[3,30],datafram:[2,3,8,27,28],datapath:10,dataset:[1,4,5,7,8,9,10,11,15,16,21,22,23,24,25,27,30,31,32],dataset_dir:15,dataset_id:[3,16,20,22,24,29,31],dataset_nam:30,dataspec:2,datatyp:1,date:[22,28,30],datetim:[22,30],dateutil:30,datset:10,daunt:25,db:[4,9,15,16,24,29],db_delta:4,de:[9,22],declar:8,decod:1,decor:[13,20],deep:[23,27,30],def:[15,30],defin:[10,30],delai:9,demonstr:[22,23],densiti:3,densiyi:5,depend:[25,29],dereverb:[4,9,11,24,31],dereverber:[9,11],dereverberate_jit:9,desc:13,descend:3,descript:[5,8,9,13,28],design:24,desir:[1,9,11],desktop:25,dest_dir:15,destin:15,detail:[4,22,27,30],detect:[7,26],dev:26,deviat:9,devic:[10,30],dict:[2,10,11,13,14,15,30],dictionari:[2,3,10,11,13,14,15,20,28],dictlist:13,dictlist_to_dict:13,did:10,diff:28,differ:[3,16,23,29,30],differenti:[10,23],difficult:31,dilation_kern:9,dimens:11,dimension:[3,8,25],dir:[3,10,15,16,22,24,25,29,30,31],directli:[5,7],directori:[3,10,14,15,16,18,22,25,30],discret:[22,25],disk:[3,15,27],displai:3,distanc:8,distinct:27,distribut:[3,5,9,18,25],divid:4,dname:15,doc:[8,10,24,26,29],docstr:3,document:[8,22,26,32],doe:26,doesn:[3,15,16],don:[16,22,29],done:[22,24,27,31],downstream:3,draw:30,drive:29,drop:3,drop_zero_len_unit:10,dt:30,dtype:28,durat:[3,5,10,11,20,28,30],dure:4,e:[3,10,11,13,15,16,20,21,22,25,26,29],each:[3,4,7,8,11,13,22,24,27,28,30],eas:[24,29],easi:23,easier:[25,29,30],easili:[18,23,24,26],echo:[9,22],echo_rang:9,echo_reduct:9,edit:[18,26],eecut:21,effici:15,either:[5,7,8,22,30],element:[13,22],elifesci:31,els:[3,15],emb:1,embed:8,embedd:1,embeddable_imag:1,emploi:18,empti:29,enabl:[16,24],end:[3,20,24],end_tim:20,energi:[5,7],enough:[3,22,25],ensur:20,ensure_ascii:15,entir:[27,30],env:26,envelop:9,environ:[25,26],epoll1:21,erosion_kern:9,error:[20,21,25],especi:31,estim:[3,5,24],etc:[7,9],even:16,event:6,ever:29,everi:[11,23],everyth:22,exactli:25,exampl:[3,4,10,15,16,21,22,23,24,25,26,27,28,29,30,31,32],except:15,exclud:10,exclus:3,execut:3,exist:[3,15,16,20,22,24,26,28,29],exp:25,expect:3,explain:32,explor:[0,3,5,27],ext:[24,30],extend:[2,10,15,30],extens:[16,21,30],extern:[15,16,29],external_data:29,extra:30,extract:[7,10,11,22,28,30,32],extract_window:11,f:[13,20,24,28,31],f_exist:20,factor:[3,5,8,13],fail:[3,31],fairi:24,fairli:23,fals:[1,3,4,5,7,8,11,13,14,15,16,22,24,28,31],familiar:[25,28],familiaris:22,famous:24,faq:32,faster:[26,32],favourit:24,fb8072:3,fccde5:3,fdb462:3,featur:[7,10,22,32],feedback:2,few:22,fewer:21,ffed6f:3,ffffb3:3,ffrom:7,fft_rate:4,fft_size:[4,31],field:[2,10,16],fieldrecord:16,figur:[16,29],file:[3,10,11,14,15,16,18,20,22,24,25,28,29,31],file_list:15,fileexistserror:[3,14,16],filenotfounderror:16,filepath:30,files:30,files_to_seg:[24,30],filter:16,find:[10,25,27,29,30],find_unit:10,fine:[23,31],first:[3,4,10,14,21,22,26,28,29,30,31],fix:[15,16,21],flatten:13,flatten_list:13,flatten_spectrogram:11,flexibl:[28,30],focal:10,folder:[3,10,14,15,16,25,29,30],follow:[21,25,27,31],foolproof:18,forc:16,forg:26,fork:21,format:[10,28,30],found:[3,16,22,24,31],frame:[1,4,9,11],freez:21,frequenc:[3,4,5,7,9,10,11,24,30],fresh:26,friendli:30,from:[1,3,4,5,7,8,9,10,11,13,14,15,16,18,22,23,24,25,27,29,30,31],full:[3,5,11,30],funtion:3,further:24,g:[3,10,11,13,15,16,20,21,22,25,29],gain:30,gauss_sigma:[4,9],gaussian:[4,9],gaussian_blur:9,gcc:21,gener:[1,3,4,10,12,14,15,16,23,24,26,28],georg:15,get:[3,10,11,16,21,24,28,29,31],get_chunk:13,get_file_path:[16,24,30,31],get_indv_unit:11,get_indv_units_parallel:11,get_mark:2,get_mean_sd_mfcc:7,get_metadata:[10,30],get_norm_spectral_envelop:9,get_peak_freq:7,get_segment_info:10,get_unit:[3,22,24,27],get_unit_spectrogram:[11,15,24],get_vocalisation_unit:11,get_wav:[10,24],get_wavs_w_annot:[16,24,30,31],gif:22,git:[13,26,29],github:[26,27],given:[10,11,15,16,24],go:[13,22,25],good:[3,31],gpu:[8,18,25],gre:25,great:22,great_tit:[16,22],greatli:24,grei:1,group:[3,8,15,30],grouping_label:2,grpcio:21,guid:[18,30,31],gz:[15,25],gzip:14,ha:[3,14,16,22,25,27],half:25,happen:3,happi:22,have:[3,16,21,22,24,25,26,27,28,29,30,31],hdbscan:[3,8,21],hdbscan_clust:8,hdd:16,head:[22,28],heard:24,help:[18,25,26],here:[22,24,30,31],hertz:[9,10],high:32,highcut:[4,24,31],highli:[22,26],hint:20,histogram:[3,5],hold:9,home:[16,30],hood:26,hop:[9,11],hop_length:[4,9,11,24,31],hop_length_m:4,hopefulli:25,hover:22,how:[4,8,9,16,22,23,25,30,31],howev:31,hpc:18,html:[2,24],html_marker:2,http:[13,15,24,26],hz:[4,9,11,30],hz_to_mel_lib:9,i:[3,10,12,22,24,25,26,27,28,29,30,31],ib:30,id:[1,3,8,10,11,14,15,20,24,28,30],identifi:10,idx:8,ignor:[10,30],ignore_check:16,ignore_label:10,imag:[1,15,26],immut:16,implement:[8,11,15,26],in_dir:25,includ:[4,5,15,16,27,29,30],incorpor:10,indent:15,index:[3,9,10,14,15,22,24,30,31],indic:[4,5],individu:[2,3,4,14,15,22,24,30],indv_list:2,info:[13,21,22,24,31],inform:[3,10,13,20,22,24,28,29,30],inherit:20,init:25,initialis:16,input:20,inset:10,instal:[21,30,32],instanc:[5,10,16,20,22,24,28,31],instance_of:30,instanti:[3,20],instead:8,instruct:[18,27],int64:28,integer_format:10,intend:[16,18,30],intens:25,inter:28,interact:[0,1,2,3,22,25,32],interactib:1,interest:[20,22,30],interfer:26,interpol:9,interpret:25,interv:28,invert:1,involv:25,io:[13,24,25,30],ioi:28,ip_head:25,ipython:21,is_list_of_int:20,is_list_of_str:20,isft:30,issu:[27,32],item:[4,13],itemsview:15,iter:[10,11,13,15,20],iterable_nam:13,its:[3,5,7,10,11,16,22,23,29,30,32],itself:[9,24],jit:11,jitter:5,jmv6r:13,job:[18,25],jpg:15,js:9,json:[3,10,14,15,16,20,25,28,30],json_fil:3,json_loc:15,json_object:15,json_outdir:[10,30],jsonencod:15,just:[13,16,24,29,30],kantodata:[1,3,4,5,7,8,9,10,11,13,15,16,21,22,24,25,27,31,32],karlb:15,kbp:30,keep:[3,4,16,23],kei:[3,5,7,10,11,14,15,24,31],kernel:[3,4,5,9,21],kernel_s:9,keyerror:4,keyword:[3,5,10],khz:30,kib:30,kind:30,known:[26,32],kwarg:[3,4,5,8,10,13,20],lab:2,label:[0,1,2,3,8,10,15,20,22,27,30,32],labels_:8,labels_to_ignor:[10,24,30],lachlan:[9,22],lambda:28,languag:25,larg:[0,3,10,25,29],larger:[8,30],last:[3,4,13,24],last_chunk:[13,20],last_not:15,last_offset:28,later:30,latest:24,latter:31,launch:[19,27],lavf57:30,learn:[22,23,27,30,31],least:25,legend:[2,5],legibl:13,leland:1,len:[7,13],len_iter:[13,20],lenght:[1,3,5,11],length:[3,4,5,9,10,11,13,22,24,28],length_:20,let:[15,22,24,29,30],level:[24,25,27],librari:[24,25],librosa:24,lightn:23,lightweight:27,lightweigth:3,like:[3,8,15,16,21,22,24,25,26,28,29,30],liken:24,limit:3,link:[15,16],link_project_data:[16,29],list:[1,2,3,5,7,10,11,13,14,15,16,20,28],liter:3,littl:[24,25],live:[16,29],ll:27,load:[1,3,11,15,16,22,28,30,31],load_app_data:1,load_dataset:[15,24,28],loc:31,local:[22,24,31],locat:[3,10,11,16,29],locklei:24,log:[7,18,25],logfil:25,longer:3,look:[16,22,24,25],lose:3,loss:9,lot:[3,25,31],lowcut:[4,24,31],lower:21,lower_freq:20,lowercas:[16,29],lst:13,luscinia:22,m:[5,18,25],machin:[3,16,21],magic:21,magnu:24,main:[3,20,24],major:22,make:[15,25,29,30,32],make_tarfil:[15,25],makedir:[15,30],male:[22,24],manag:13,mani:[4,9,22,25],manifold:8,manipul:11,manual:[30,31],map:2,marker:2,marker_typ:2,mask:[4,5],match:10,matplotlib:5,matrix:8,max:[9,10,24],max_amplitud:[10,20,22],max_db:4,max_lenght:[5,31],max_n_lab:3,max_unit_length:4,maxfreq:[3,5,24],maximum:[3,4,5,7,11,24],mcinn:1,me:25,mean:[7,27],mel:[3,4,5,7,9],mel_bin:9,mel_spectrogram:9,mel_to_hz:9,mels_to_hz:9,melscal:7,melspectrogram:[3,5,11],mem:25,membership:3,memori:[3,25],merino:[22,28],messag:21,meta:30,metadata:[3,10,14,15,16,20,28],metadata_dir:10,method:[3,4,9,15,22,24,27,28,30,31],mfcc:7,might:[16,21,22,24,25,30,31],million:25,min:[9,10,24],min_amplitud:[10,20,22],min_cluster_s:[8,22],min_dist:8,min_dur:[10,24,30],min_freqrang:[10,24,30],min_level_db:9,min_sampl:[3,8,22],min_silence_length:4,min_unit_length:4,mindb:9,minfreq:[3,5,24],minim:[16,22],minimum:[3,4,5,7,8,10,11,24],minmax:9,minmax_freq:9,minut:24,miss:3,mk_colour:2,mkdir:[16,24,29,30,31],ml:32,mode:26,model:[2,25,27,30,31,32],modif:18,modifi:[4,13,15,28],modul:[0,6,12,14,17,20,31],monkei:30,monoton:24,more:[4,8,18,22,24,25,27,29,30,32],morpholog:10,most:[25,28,30],motiv:32,move:[3,16],mspaced_mask:5,much:[4,9,24,26],multi:[18,25],multipl:[11,24,25],must:[3,5],my:22,myproject:16,n:[1,4,5,7,13,25,26,31],n_chunk:[13,20],n_compon:8,n_featur:8,n_fft:[14,24],n_mfcc:7,n_neighbor:8,n_sampl:8,n_song:3,n_worker:[13,20],name:[2,3,4,10,14,15,16,28,29,30],natur:24,nbin:[3,5],ndarrai:[1,2,5,7,8,9,10,11,15],necessari:[20,26],necessarili:31,need:[21,22,24,25,26,27,30],nest:15,network:[3,31],neural:[3,31],never:29,new_attr:16,new_dataset:3,new_project:16,new_valu:16,newli:[3,31],next:[22,29],nilo:[18,22,28],nilomr:[23,26,30],node:[18,25],nois:[3,8,10,22,24,30],noisi:30,non:30,none:[1,3,4,5,7,8,10,11,13,15,16,20,30],norm:9,normal:[10,25,30],normalis:9,note:[3,15,24,30],now:[11,22,24,29,30,31],nox:26,np:[1,2,5,7,8,9,10,11,15,28],nparray_dir:11,nparray_or_dir:5,num_cpu:[1,3,4,8,10,11,21],num_mel_bin:[4,24,31],numba:[11,25],number:[0,3,4,5,7,8,10,13,21,29],numpi:[1,2,5,7,8,9,10,11,15],numpyencod:15,nvidia:[25,26],o:[12,15],obj:15,obj_id:13,object:[4,5,7,8,9,10,11,13,14,15,16,21,22,27,30,31],offset:[3,5,10,11,14,15,22,24,28,30,31],often:24,old:16,oldham:24,onc:[3,22,24,27],one:[1,3,4,5,10,11,15,16,22,24,25],ones:[3,16,28],onli:[10,16,21,22,25,26],onset:[3,5,10,11,14,22,24,28,30,31],onsets_offset:[5,10],open:[3,11,27],open_gzip:14,open_label_app:[3,22,27],oper:13,optim:25,optimis:[13,25],option:[1,2,3,4,5,7,8,9,10,11,13,14,15,16,31],order:[3,7,26],org:[24,31],organis:[3,23],origin:[3,9,15,16],os:[16,25],oset:15,other:[3,5,10,16,20,22,24,27,28,29],otherwis:29,our:30,out:[3,18,21,25],out_dir:[18,24,25],outlier:[3,5],output:[3,5,10,14,15,18,23],output_filenam:15,outsid:26,over:[5,24,25],overflow:15,overhead:25,overkil:25,overlai:[3,5],overlap:14,overwrit:[3,16,22,24,27],overwrite_data:[3,24,31],overwrite_dataset:[3,22,24,31],overwrite_json:14,own:[3,22,23,27,30],oxford:18,oxfordshir:22,p:25,packag:[22,25,26,30,32],pad:[3,11,14],pad_length:11,pad_spectrogram:11,page:27,pair:10,palett:[2,3,5],panda:28,paralel:[13,21,30],parallel:[1,3,4,8,10,11,13,18,25],parallelis:[13,25],param:[22,24,25,28,31],paramet:[1,2,3,5,7,8,9,10,11,13,14,15,16,21,22,24,25,27,28,31],parent:[15,16,24,30,31],pars:[2,10,14,18,30],parse_boolean:2,parse_sonic_visualiser_xml:[10,14,24,30],parser:[12,30],parser_func:[10,24,30],particular:[23,25],paru:22,pass:[3,4,5,8,10,15,25,29],patch:30,path:[1,3,5,10,11,14,15,20,22,24,30,31],pathlib:[3,10,11,14,15,16,20,24,29,30,31],pathlik:16,pbar:[10,14],pcm:30,pcm_16:10,pd:[2,8],peak:7,peng:[18,25],peopl:30,per:[1,3,4,11],percentag:[2,5,7],perform:[8,9,32],petrel:[16,24],phrase:24,pickl:[11,21],pictur:30,pid:22,pip:26,pkg_resourc:[24,30,31],platform:[24,25],pleas:27,plot:[1,2,3,7,22,24,31],plot_exampl:3,plot_summari:3,plu:24,png:1,point:[3,8,16,22,29],poll:21,popul:[3,22],popular:25,possibl:[5,25,26],power:[7,23],precis:24,predict:30,prefer:26,prepar:[1,2,3,22,31,32],prepare_datasourc:1,prepare_datasource_parallel:1,prepare_interactive_data:[3,22,27],prepare_legend:2,present:[3,10,11,15,20,22,23],preserv:30,pretti:13,previou:31,print:[3,10,13,25,28,29,30,31],print_dict:13,print_parallel_info:13,probabl:[25,29],problem:16,process:[1,5,6,13,19,21,22,24,27,32],produc:3,progress:[10,13],projdir:[3,10,15,16,20,24,29,30,31],project:[3,8,10,15,16,20,22,23,24,26,30,31,32],project_data_dir:16,project_root:[29,30,31],projroot:16,promot:29,prompt:3,properli:30,properti:2,provid:[3,4,5,7,10,16,20,23,24,25,29,30,32],pty:25,pur:24,purpos:20,purr:24,py:[22,24,25,31],pycharm:22,pykanto:[21,22,23,24,27,29,30,31,32],pykanto_data:[16,22,30],pyrigh:20,pytest:26,python:[25,26],pytorch:[23,26],queri:[3,28],quickli:[3,5,9,25],r:[25,30],rai:[13,17,18,19,21,22,24,25,30,31],raid:29,rais:[3,4,5,14,15,16],ram:21,rand_jitt:5,random:[3,5,25],random_subset:[3,25],rang:[2,3,5,9,10,30],rapid:26,rapidsai:26,rate:[4,10,11],ratio:[9,10],raw:[10,16,22,24,25,30,31],raw_data:[16,20,24,29,30,31],raw_data_dir:10,re:30,reach:4,read:[10,14,15,24,29,30],read_json:15,readi:[1,24,29,30],readwav:[10,24,30],readwav_patch:30,real:[22,25],realiti:30,reallist:3,reason:[22,31],rec_unit:30,recald:[18,22,28],recommend:[22,26,29],record:[10,22,31,32],recordist:[22,28],recurs:16,redis_password:25,reduc:[4,8,9,10,25],reduce_and_clust:8,reduce_and_cluster_parallel:8,reduct:[3,8,9,25],refer:[11,18,25,30,31],region:[20,30],regular:8,regularli:5,reilli:15,rel:9,relat:[6,20],releas:26,relev:[10,14,30],relink_data:15,reload:[3,22,27],remain:[2,10],remaining_indv:2,rememb:3,remot:29,remov:[10,26],repeat:24,repo:29,report:[16,29],repositori:[23,29],repres:25,represent:[3,11,22,27],reproduc:[24,32],request:25,requir:[3,9,16,21,22,27,30],resampl:[10,24,30],rescal:9,research:[10,24],resort:31,resourc:[3,16,25,29],resource_filenam:[24,30,31],respect:3,rest:16,restart:21,result:[3,5,7,10,11,24,29],retriev:[15,30],retrieve_spectrogram:[11,24],return_kei:3,return_path:15,reus:23,reverber:4,rgb:1,rifftag:30,right:25,robb:24,robert:[9,22],roll_perc:[5,7,24],root:[16,29],root_dir:16,row:[4,22,28],run:[3,18,21,22,23,24,25,26,27,28,30],runtim:25,s:[2,3,7,9,10,11,13,16,20,21,22,24,26,29,30],safe:15,safeti:24,sai:[25,30],sainburg:[10,15,22],same:[1,3,11,16,21,25,26,29,30],sampl:[3,4,8,9,10,11,16,22,30],sample_info:[3,28],sample_r:[20,30],sample_s:[3,5],save:[1,3,10,11,15,18,21,27,28,30],save_json:15,save_melspectrogram:[10,11],save_seg:10,save_song:15,save_subset:15,save_to_disk:[3,28],save_to_json:15,scale:7,scatterplot:2,schedul:[17,19,25],scienc:22,scikit:26,score:23,scratch:30,script:[18,23,25],sd:7,sdata:2,seaborn:5,search:[16,30],search_parent_directori:29,sec:24,second:[3,4,10,11,14,25,30],section:[29,31],see:[3,8,10,16,18,21,22,25,26,27,28,29,30,31],seekabl:10,segment:[3,4,5,6,11,14,16,20,22,24,25,29,32],segment_fil:[10,30],segment_files_parallel:[10,24,30],segment_into_unit:[3,22,24,25,27,31],segment_is_valid:10,segment_song_into_unit:10,segment_song_into_units_parallel:10,segmentannot:[10,14,20,30],segmentmetadata:[10,30],segmentmetadata_patch:30,segmet:10,select:[5,24],selector:26,self:[3,8,15,30],separ:[4,10,15,24],seri:[10,24,28],serializ:[15,20],server:29,session:25,set3_12:3,set:[2,3,4,9,21,22,23,24,28,30,31,32],set_rang:2,setup:22,sf:10,sh:25,shape:8,share:16,shorter:[3,13,30],should:[3,22,25],show:[3,7,22,24],show_extreme_song:[3,5],show_minmax_frequ:5,show_spec_centroid_bandwidth:5,sick:24,sigma:4,signal:[24,30],silenc:[3,4,20],silence_dur:22,silence_threshold:[4,31],similar:23,simpl:[22,25],simpli:[22,27,29,31],simplic:22,singl:[10,11,24,25,28,31],size:[2,3,4,8,13,22],skip:3,skipkei:15,slaunch:[18,25],slice:25,slightli:[25,31],slow:25,slurm:25,small:[3,8,22,25],smaller:[3,22,29,30],snr:31,sns_histoplot:5,so:[3,24,25,30],softwar:32,some:[24,25,26,28,30,31],someth:25,song:[1,3,9,11,15,20,22,23,24,29,30,31,32],song_level:[1,3,4,8,11,24,27,28],sonic:[10,14,30,31],sort:9,sort_kei:15,sound:[24,27,28],soundfil:10,sourc:[1,2,3,4,5,7,8,9,10,11,13,14,15,16,18,20,25],source_datetim:22,source_dir:15,source_wav:20,space:5,span_mk_siz:2,spec:[3,5,7,15,24],spec_bw:5,spec_centroid_bandwidth:[7,24],spec_length:[1,3],speci:[22,31],specif:[10,31],specrogram:3,spectral:[5,7,9,24],spectrogram:[1,3,4,5,6,7,9,10,15,16,22,24,25,29,31],sphinx:26,spot:[3,5],sr:[4,9,10,11,24,31],srun:25,stack:15,stackoverflow:[13,15],standard:[9,10,25,30],standardis:29,start:[3,20,22,24,30,31],start_tim:[20,30],state:30,std:7,step:[4,22,23],still:[25,30],storag:20,store:[4,5,15,16,20,24,29,30],storm:[16,24],str:[1,2,3,5,7,8,10,11,13,14,15,16,20,24,27,30],strang:24,strategi:21,streaminfo:30,string:[2,5,7,14,30],strongli:29,structur:[3,10,16,31],studi:22,subclass:15,subdirectori:16,subfold:10,submit:[18,25],submit_job:18,submodul:18,subset:[3,4,15,25,28],subtract:9,success:[4,13],sudo:21,sumbiss:25,summari:10,support:[15,21,26],sure:30,sw83:28,sy:25,syllabl:[14,31],symlink:16,system:[16,25,26,29],t:[3,15,16,21,22,29],tab:[3,27],tag:30,take:25,taken:4,tar:[15,25],tarfil:15,target:16,task:25,tech_com:22,tell:[24,29,30],tend:[4,30],term:31,termin:25,test:[3,15,16,19,23,26],test_dir:15,text:2,than:[18,21,25,30],thei:[3,16,22,24,25,26,29,30],them:[14,22,27,29,30,31],thi:[3,5,7,10,13,14,15,16,18,20,21,22,23,24,25,26,27,28,29,30,31,32],thing:15,think:30,those:[24,30],three:28,threshold:[4,7,9],through:[23,27],tidi:16,tim:[10,15,22],time:[3,10,13,22,24,25,30,31],timeaxisconvers:14,timedelta:30,timer:13,timestamp:3,timezon:22,tit:22,titl:[5,24],to_csv:[3,28],to_export:15,to_iter:13,to_list:28,todo:10,togeth:[23,29],too:[3,8,22],took:24,tool:[6,12,25,31],top:[4,25],top_db:[4,24,31],torchvis:26,total:[13,28],touch:29,tqdm:13,train:[3,15,25,27,30,31,32],train_dir:15,transfer:16,transform:[10,30],translat:25,tree:[16,29],trim:3,troubl:3,truli:25,tune:[23,31],tupl:[1,2,3,4,5,7,8,9,10,11,14,16],tutori:16,tweetynet:31,two:[3,14,22,25],type:[1,2,3,5,7,8,9,10,11,12,13,14,15,16,22,27,30],typeerror:15,ugli:30,ujson:15,uk:22,umap:[3,8,26],umap_:8,umap_i:8,umap_reduc:8,umap_x:8,under:[10,16,26],uniform:8,uniniti:25,uniqu:28,unit:[1,3,4,5,8,10,11,14,15,22,24,25,27,28],unit_app_data:1,unit_dur:[22,28],univers:18,unix:16,unless:13,unsupervis:3,up:[22,32],updat:[2,4,15,16,21,24,27],update_feedback_text:2,update_json_loc:16,upper_freq:[20,22],us:[1,3,4,5,8,9,10,11,13,15,16,17,18,21,23,24,26,27,29,31,32],user:[2,4,13,16,20,25,28],userwarn:[22,24],usual:[15,30],utc:[22,30],util:[10,22,24,25,29,30,31],v100:25,v:15,vagu:31,valid:[4,8,10,16,20,30],validdir:20,valu:[1,8,9],value_count:28,valueerror:[3,5,16],vari:24,variabl:[3,5,16],vector:8,verbos:[3,4,8,10,13,16,24],veri:[3,22,28,31],version:[3,8,9,10,16,26,29],via:26,view:[22,24,31],virtual:26,visualis:[3,10,14,30,31],voc_app_data:1,vocal:[3,28],vocalis:[0,1,3,4,5,7,8,10,11,15,22,24,27,28,32],vocalisation_kei:8,vocalseg:[10,22],voic:4,vscode:22,wa:[11,30],wai:[13,18,23,24,25,29,30],walk:23,want:[16,22,24,25,26,27,29,30,31],wav:[3,10,11,16,24,30],wav_dir:[10,30],wav_fil:[16,20,24],wav_filepath:[16,24,30],wav_object:24,wav_out:10,wav_outdir:[10,30],wave:30,waveaudioformat:30,wavestreaminfo:30,wavfil:[10,16,24],we:[22,24,30,31],web:[0,1,3,27],websit:32,well:[4,10,22,31],were:15,wether:[4,16],what:[3,10,13,22,30],when:[3,16,21,27],where:[3,10,13,16,24,25,29],wherev:29,wheter:[5,10],whether:[1,3,4,7,8,10,11,15,16],which:[3,5,10,15,16,18,22,23,24,25,27],whose:20,why:29,wil:29,window:[4,11,16],window_length:[4,24,31],window_offset:14,with_pbar:13,within:[10,13,16,18,25,31],without:[11,21],wlength:11,wood:22,work:[4,10,16,18,22,25,27,30,31,32],worker:[21,22,24,31],workflow:[27,32],working_tree_dir:29,world:25,would:[10,22,25,26,28,30],wrapper:[8,13,24,25],writabl:13,write:[3,22,30],write_to_json:[3,28],wrong:15,wytham:22,x11:25,x:[2,9,11,28],xml:[10,14,24],xml_filepath:[14,24,30],y:[2,11],year:10,yield:[11,13],you:[3,5,7,10,15,16,18,21,22,23,24,25,26,27,28,29,30,31],your:[3,5,7,10,16,21,22,23,24,26,27,30,31],yourself:22,zero:10,zhenghao:[18,25]},titles:["pykanto.app","pykanto.app.data","pykanto.app.main","pykanto.dataset","pykanto.parameters","pykanto.plot","pykanto.signal","pykanto.signal.analysis","pykanto.signal.cluster","pykanto.signal.filter","pykanto.signal.segment","pykanto.signal.spectrogram","pykanto.utils","pykanto.utils.compute","pykanto.utils.custom","pykanto.utils.io","pykanto.utils.paths","pykanto.utils.slurm","pykanto.utils.slurm.launch","pykanto.utils.slurm.tester","pykanto.utils.types","FAQs &amp; known issues","Basic workflow","Training ML models","Acoustic feature extraction","High Performance Computing","Installing <code class=\"docutils literal notranslate\"><span class=\"pre\">pykanto</span></code>","Interactive app","The KantoData dataset","Setting up a project","Preparing long recordings","Segmenting vocalisations","Modules"],titleterms:{"1":29,"2":29,"long":30,The:28,acceler:26,acoust:24,an:22,analysi:7,app:[0,1,2,27],area:25,attribut:28,avoid:26,basic:[22,26],cluster:[8,25],code:25,common:28,comput:[13,25],custom:[14,30],data:[1,25,29],dataset:[3,28,29],depend:[21,26],deriv:29,develop:26,directori:29,exist:30,extract:24,faq:21,featur:24,field:30,file:30,filter:9,first:25,freez:29,gpu:26,guid:32,hell:26,high:25,hpc:25,id:22,instal:26,instruct:25,interact:27,introduct:25,io:15,issu:21,kantodata:28,known:21,launch:18,librari:26,link:29,local:25,machin:25,main:2,metadata:30,ml:[23,26],model:23,modul:32,note:[22,27,29,31],onli:29,oper:28,paramet:4,path:[16,29],perform:25,plot:5,prepar:30,programmat:29,project:29,pykanto:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,26],raw:29,record:30,segment:[10,30,31],set:29,signal:[6,7,8,9,10,11],slurm:[17,18,19],spectrogram:11,storag:25,system:21,test:25,tester:19,tip:[22,25,26,29],train:23,type:20,unit:31,up:29,upload:25,us:[22,25,28,30],user:32,util:[12,13,14,15,16,17,18,19,20],vocalis:31,work:29,workflow:22,xml:30,your:[25,29]}})