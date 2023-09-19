#FROM python:3.9-slim
FROM pytorch/pytorch
#FROM projectmonai/monai:1.2.0

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install -U pip
#RUN python -m pip install SimpleITK
#RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user requirements_docker.txt /opt/app/
RUN python -m pip install -r requirements_docker.txt


#ENV INPUTS_PET="/inputs/images/pet/"
#ENV INPUTS_CT="/inputs/images/ct/"
#ENV OUTPUTS="/output/images/automated-petct-lesion-segmentation/"
#ENV NII_PATH="/opt/app/nifti"


COPY --chown=user:user . /opt/app/sw_interactive_segmentation/
WORKDIR /opt/app/sw_interactive_segmentation/
 
#CMD python src/test.py -i /input/images/pet/ -o /output/images/automated-petct-lesion-segmentation/ --use_scale_intensity_range_percentiled --non_interactive -a --disks --gpu_size small --eval_only --limit_gpu_memory_to 0.66 --resume_from checkpoint.pt
CMD python src/test.py -i /input/images/pet/ -d /tmp -o /output/images/automated-petct-lesion-segmentation/ --use_scale_intensity_range_percentiled --non_interactive -a --disks --gpu_size small --eval_only --resume_from "." -ta --dataset AutoPET2_Challenge --dont_check_output_dir --no_log --dont_crop_foreground --sw_overlap 0.75 --no_data --val_sw_batch_size 8
#CMD python src/test.py -i /input/autopet -d /tmp -o /output/images/automated-petct-lesion-segmentation/ --use_scale_intensity_range_percentiled --non_interactive -a --disks --gpu_size small --eval_only --limit_gpu_memory_to 0.66 --resume_from 195.pt -ta --dataset AutoPET --dont_check_output_dir --no_log --dont_crop_foreground --sw_overlap 0.75 --no_data -t 2
#ENTRYPOINT [ "python", "-m", "process"]
