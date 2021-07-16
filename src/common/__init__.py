from tqdm import tqdm


def training_progress(loader, epoch, epochs, loss, debug):

    return tqdm(loader, desc="Running Epoch {:03d}/{:03d}".format(epoch + 1, epochs),
                leave=False, ncols=117, unit='step', unit_scale=True,
                postfix={"loss": "{:.3f}".format(float(loss))}) if debug else loader


def testing_progress(loader, epoch, epochs, debug):

    return tqdm(loader, desc="Testing Epoch {:03d}/{:03d}".format(epoch + 1, epochs),
                leave=False, ncols=117, unit='step', unit_scale=True) if debug else loader


def building_progress(df, debug):
    return tqdm(df.iterrows(), desc='building', total=len(df),
                leave=False, ncols=117, unit='review', unit_scale=True) if debug else df.iterrows()
