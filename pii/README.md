# PII Removal

I did this using c7a.4xlarge EC2 instance (c8g would be cheaper and faster but they ran out).

Remember to make sure that each side has the public SSH keys of the other in `/.ssh/authorized_keys`.

First I copied over the user_id to cluster mappings by running this on compute canada. Replace `EC2_DNS` with that of your EC2 instance.

```bash
# specify your EC2 DNS
EC2_DNS="??"

# include the user in one var:
REMOTE="ubuntu@${EC2_DNS}"

# Start SSH agent if not already running
eval $(ssh-agent)

# Create a master connection first
ssh -o "ControlMaster=yes" \
    -o "ControlPath=~/.ssh/cm-%r@%h:%p" \
    -o "ControlPersist=5m" \
    ${REMOTE} "echo 'Connection established'"
```

Now run this
```bash
# Now use the connection for each file transfer
for dir in processed_*; do
  # Create directory if it doesn't exist
  ssh -o "ControlPath=~/.ssh/cm-%r@%h:%p" ${REMOTE} "mkdir -p ~/${dir}"
  
  # Copy the file using the existing connection
  scp -o "ControlPath=~/.ssh/cm-%r@%h:%p" "$dir/user_clusters.json" "${REMOTE}:~/${dir}/"
done

# Close the master connection when done
ssh -O exit -o "ControlPath=~/.ssh/cm-%r@%h:%p" ${REMOTE}
```

We need at least one full partition so I also copied over the 25 clusters.
```bash
scp -r processed_25_clusters "${REMOTE}:~/"
```

Then on the EC2 instance, make sure the virtual environment is activated and installed.

```bash
cd ~/steering-sm-personas
source .venv/bin/activate
pip install -r requirements.txt
```

I first ran `merge_df.py` to merge all the 25 clusters into one cluster and break the chains into messages while keeping track of the IDs. This outputs `~/all_messages/merged_messages.parquet`.

```bash
cd ~/steering-sm-personas/pii
python merge_df.py
```

I then ran `sample.py` to get some smaller files in `all_messages/` to work with. This is not required if you are following along, and not doing development. This is simply for debugging.

```bash
python sample.py
```

Then run the pii removal in the background, will take about 2 hours.

```bash
nohup python pii_temp.py > logs/pii-2.log 2>&1 &
```

Then run `rebuild_chains.py` to get it back into a blob

```bash
python rebuild_chains.py
```

Finally run `rebuild_clusters.py` to rebuild it into `cleaned/`

```bash
python rebuild_clusters.py
```

Then scp it back to compute canada, but to `scratch`. Run this on EC2 as usual.

```bash
cd ~/
# Create a master connection first
ssh -o "ControlMaster=yes" \
    -o "ControlPath=~/.ssh/cm-%r@%h:%p" \
    -o "ControlPersist=5m" \
    jchooi@narval.alliancecan.ca "echo 'Connection established'"
```

Then copy it
```bash
ssh -o "ControlPath=~/.ssh/cm-%r@%h:%p" jchooi@narval.alliancecan.ca "mkdir -p ~/projects/ctb-liyue/s4yor1/pii_removed/"
scp -o "ControlPath=~/.ssh/cm-%r@%h:%p" -r cleaned/processed_* jchooi@narval.alliancecan.ca:~/projects/ctb-liyue/s4yor1/pii_removed/
```


## Notes

Here are the entities that are removed (and their corresponding token)
```
<CREDIT_CARD>
<CRYPTO>
<EMAIL_ADDRESS>
<IP_ADDRESS>
<PHONE_NUMBER>
<URL>
```

These are specified in `conf.json`.

I removed privatization for @username.bsky.social (so it's still in the dataset). 
